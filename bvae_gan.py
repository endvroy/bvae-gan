import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils


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
args.lmbda = 1

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


# Create the generator
vae_model = VAE().to(device)
gen_net = Generator().to(device)
disc_net = Discriminator().to(device)

# optimizers
vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
disc_optimizer = optim.Adam(disc_net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
gen_optimizer = optim.Adam(gen_net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

bce_loss = nn.BCELoss()


def calc_vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def bvae_gan_train():
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)

            # encode data with VAE
            vae_optimizer.zero_grad()
            z, mu, logvar = vae_model(data)
            z = z.reshape(z.shape + (1, 1))
            # the vae loss is part of the generator loss, so no torch.no_grad
            recon_batch = gen_net(z)
            vae_loss = calc_vae_loss(recon_batch, data, mu, logvar)
            vae_loss.backward()

            # Train discriminator with real data
            disc_net.zero_grad()
            # Format batch
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            disc_output = disc_net(data).view(-1)
            # Calculate loss on all-real batch
            disc_loss_real = bce_loss(disc_output, label)
            disc_loss_real.backward()
            # Calculate gradients for D in backward pass
            D_x = disc_output.mean().item()

            # Train discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = gen_net(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            disc_output = disc_net(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            disc_loss_fake = bce_loss(disc_output, label)
            disc_loss_fake.backward()
            # Calculate the gradients for this batch
            D_G_z1 = disc_output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            disc_loss = disc_loss_real + disc_loss_fake

            # Update D
            disc_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gen_net.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Sigan_args.nce we just updated D, perform another forward pass of all-fake batch through D
            gen_output = disc_net(fake).view(-1)
            # Calculate G's loss based on this output
            gen_loss = bce_loss(gen_output, label)
            # Calculate gradients for G
            gen_loss.backward()
            # for logging only
            D_G_z2 = gen_output.mean().item()
            # Update G
            gen_optimizer.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, batch_idx, len(data_loader),
                         disc_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(gen_loss.item())
            D_losses.append(disc_loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (batch_idx % 500 == 0) or ((epoch == args.epochs - 1) and (batch_idx == len(data_loader) - 1)):
                with torch.no_grad():
                    fake = gen_net(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


if __name__ == '__main__':
    bvae_gan_train()
