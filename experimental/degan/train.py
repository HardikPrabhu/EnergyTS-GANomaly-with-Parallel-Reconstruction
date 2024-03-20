import torch
import json
import torch.nn as nn
import random
from model import Generator,ConvDiscriminator


# setting up cuda and replication
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def weights_init(m):
    """
    A function to initialize model weights
    """
    classname = m.__class__.__name__
    if classname.find('Conv1D') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    print(device)
    # Load config
    with open('../../config.json', 'r') as file:
        config = json.load(file)

    prefix = "../../"

    # Training configs
    nz = config['training']['latent_dim']
    window_size = config['preprocessing']['window_size']
    num_epochs = config['training']['num_epochs']
    w_gan_training = False  # False always
    batch_size = config['training']['batch_size']
    lrG = config['training']['lrG']  # learning rates for both
    lrD = config['training']['lrD']
    betaG = config['training']['betaG']
    betaD = config['training']['betaD']


    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    # Create the generator
    netG = Generator(nz, window_size)
    netG.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = ConvDiscriminator(window_size)
    netD.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # data

    X_train = torch.load(f"X_train_{b_id}.pt")
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                             shuffle=True)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(betaD, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(betaG, 0.999))

    # training:
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0
    # for image generation (while training)
    fixed_noise = torch.normal(0, 1, size=(16, nz), device=device)
    print("--GAN--")
    print("Starting Training Loop...")
    real_label = 1.
    fake_label = 0.
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for step, data in enumerate(dataloader):
            real_cpu = data.to(device).float()
            b_size = real_cpu.size(0)

            # train netD
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train netG
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            netG.zero_grad()

            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, step, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    torch.save(netD, f'gan_netD_{b_id}.pth')
    torch.save(netG, f'gan_netG_{b_id}.pth')


