import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from utils import SGHMC, Gibbs
import itertools


class BVAEGANArgs:
    pass


args = BVAEGANArgs()

args.cuda = torch.cuda.is_available()
# Learning rate for optimizers
args.lr = 0.0002
# Beta1 hyperparam for Adam optimizers
args.beta1 = 0.5
# Size of z latent vector (i.e. size of generator input)
args.nz = 100
# Size of feature maps in generator
args.ngf = 64
# Number of channels in the training images. For color images this is 3
args.nc = 1
# Size of feature maps in discriminator
args.ndf = 64
args.epochs = 5
args.batch_size = 128
args.log_interval = 10
args.n_vae = 2
args.n_gen = 2
args.n_disc = 2
args.steps = 1

device = torch.device('cuda' if args.cuda else 'cpu')

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

# Training dataset
cuda_dl_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **cuda_dl_kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, args.nz)
        self.fc22 = nn.Linear(400, args.nz)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 4,
                               4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (gan_args.ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (gan_args.ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 2, args.ngf,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (gan_args.ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf, args.nc,
                               2, 2, 2, bias=False),
            nn.Sigmoid()
            # state size. (gan_args.nc) x 28 x 28
        )

    def forward(self, inp):
        return self.main(inp)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (gan_args.nc) x 28 x 28
            nn.Conv2d(args.nc, args.ndf, 4, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. gan_args.ndf x 16 x 16
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*2) x 8 x 8
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*4) x 4 x 4
            nn.Conv2d(args.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)


l_gen = lambda d: -d  # -torch.nn.functional.logsigmoid(d)
l_dis_real = lambda d: -d  # -torch.nn.functional.logsigmoid(d)
l_dis_fake = lambda d: d  # -torch.nn.functional.logsigmoid(-d)

# create models
vae_nets = [VAE().to(device).eval() for i in range(args.n_vae)]
gen_nets = [Generator().to(device).eval() for i in range(args.n_gen)]
disc_nets = [Discriminator().to(device).eval() for i in range(args.n_disc)]


def vae_loss_(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def calc_vae_loss(vae_net, gen_net, data):
    z, mu, logvar = vae_net(data)
    z = z.reshape(z.shape + (1, 1))
    recon_batch = gen_net(z)
    vae_loss = vae_loss_(recon_batch, data, mu, logvar)
    return torch.mean(vae_loss)


def calc_gen_loss(vae_net, gen_net, dis_net, data):
    z, mu, logvar = vae_net(data)
    z = z.reshape(z.shape + (1, 1))
    recon_batch = gen_net(z)
    vae_loss = vae_loss_(recon_batch, data, mu, logvar)
    z0 = torch.randn((args.batch_size, args.nz, 1, 1), device=device)
    fake = gen_net(z0)
    gen_loss = l_gen(dis_net(fake))
    return torch.mean(gen_loss) + torch.mean(vae_loss)


def calc_disc_loss(gen_net, dis_net, data, prior_var=1e2):
    z = torch.randn((args.batch_size, args.nz, 1, 1), device=device)
    fake = gen_net(z)
    return torch.mean(l_dis_real(dis_net(data))) \
           + torch.mean(l_dis_fake(dis_net(fake)))
    # + torch.sum(torch.stack([torch.nn.functional.mse_loss(torch.zeros_like(p), p, reduction='sum')
    #                          for p in dis_net.parameters()])) / (2 * prior_var)
    # return torch.mean(l_dis_real(dis_net(x))) + torch.mean(l_dis_fake(dis_net(gen_net(z))))


def E_vae_loss(vae_net, data, gen_nets, _disc_nets):
    x, y = data
    x = x.to(device)
    return torch.mean(torch.stack([calc_vae_loss(vae_net, gen_net, x) for gen_net in gen_nets]))


def E_gen_loss(gen_net, data, vae_nets, disc_nets):
    x, y = data
    x = x.to(device)
    return torch.mean(torch.stack([calc_gen_loss(vae_net, gen_net, disc_net, x)
                                   for vae_net, disc_net in
                                   itertools.product(vae_nets, disc_nets)]))


def E_disc_loss(disc_net, data, _vae_nets, gen_nets):
    x, y = data
    x = x.to(device)
    return torch.mean(torch.stack([calc_disc_loss(gen_net, disc_net, x) for gen_net in gen_nets]))


if __name__ == '__main__':
    Gibbs([E_vae_loss, E_gen_loss, E_disc_loss],
          [vae_nets, gen_nets, disc_nets],
          data_loader,
          args.epochs, steps=args.steps, Optim=torch.optim.Adam, device=device)

    print(gen_nets[0](torch.randn((1, args.nz, 1, 1), device=device))[0].detach())
