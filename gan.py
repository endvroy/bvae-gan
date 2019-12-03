import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils


class GANArgs:
    pass


gan_args = GANArgs()

gan_args.cuda = torch.cuda.is_available()
# Learning rate for optimizers
gan_args.lr = 0.0002
# Beta1 hyperparam for Adam optimizers
gan_args.beta1 = 0.5
# Size of z latent vector (i.e. size of generator input)
gan_args.nz = 100
# Size of feature maps in generator
gan_args.ngf = 64
# Number of channels in the training images. For color images this is 3
gan_args.nc = 1
# Size of feature maps in discriminator
gan_args.ndf = 64
gan_args.epochs = 5

device = torch.device('cuda' if gan_args.cuda else 'cpu')

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, gan_args.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

# Training dataset
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(gan_args.nz, gan_args.ngf * 4,
                               4, 1, 0, bias=False),
            nn.BatchNorm2d(gan_args.ngf * 4),
            nn.ReLU(True),
            # state size. (gan_args.ngf*8) x 4 x 4
            nn.ConvTranspose2d(gan_args.ngf * 4, gan_args.ngf * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(gan_args.ngf * 2),
            nn.ReLU(True),
            # state size. (gan_args.ngf*4) x 8 x 8
            nn.ConvTranspose2d(gan_args.ngf * 2, gan_args.ngf,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(gan_args.ngf),
            nn.ReLU(True),
            # state size. (gan_args.ngf*2) x 16 x 16
            nn.ConvTranspose2d(gan_args.ngf, gan_args.nc,
                               2, 2, 2, bias=False),
            nn.Tanh()
            # state size. (gan_args.nc) x 28 x 28
        )

    def forward(self, inp):
        return self.main(inp)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (gan_args.nc) x 28 x 28
            nn.Conv2d(gan_args.nc, gan_args.ndf, 4, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. gan_args.ndf x 16 x 16
            nn.Conv2d(gan_args.ndf, gan_args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gan_args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*2) x 8 x 8
            nn.Conv2d(gan_args.ndf * 2, gan_args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gan_args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*4) x 4 x 4
            nn.Conv2d(gan_args.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)


# Create the generator
gen_net = Generator().to(device)
disc_net = Discriminator().to(device)

bce_loss = nn.BCELoss()
# Setup Adam optimizers for both G and D
disc_optimizer = optim.Adam(disc_net.parameters(), lr=gan_args.lr, betas=(gan_args.beta1, 0.999))
gen_optimizer = optim.Adam(gen_net.parameters(), lr=gan_args.lr, betas=(gan_args.beta1, 0.999))


def gan_train():
    # For each epoch
    for epoch in range(gan_args.epochs):
        # For each batch in the dataloader
        for i, data in enumerate(data_loader, 0):
            ## Train with all-real batch
            disc_net.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            disc_output = disc_net(real_cpu).view(-1)
            # Calculate loss on all-real batch
            disc_loss_real = bce_loss(disc_output, label)
            # Calculate gradients for D in backward pass
            disc_loss_real.backward()
            D_x = disc_output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, gan_args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = gen_net(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            disc_output = disc_net(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            disc_loss_fake = bce_loss(disc_output, label)
            # Calculate the gradients for this batch
            disc_loss_fake.backward()
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
            #for logging only
            D_G_z2 = gen_output.mean().item()
            # Update G
            gen_optimizer.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, gan_args.epochs, i, len(data_loader),
                         disc_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(gen_loss.item())
            D_losses.append(disc_loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == gan_args.epochs - 1) and (i == len(data_loader) - 1)):
                with torch.no_grad():
                    fake = gen_net(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


if __name__ == '__main__':
    gan_train()
