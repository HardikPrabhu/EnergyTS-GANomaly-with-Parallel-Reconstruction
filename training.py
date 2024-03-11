import torch
import json
import torch.nn as nn
import random
from model import ConvDiscriminator, ConvGenerator
import matplotlib.pyplot as plt

"""
Model Training 
---------------
 - Script for training in two modes: a) Simple loss b) WGAN loss. 
 - Could be switched between the two by setting config['training']['w_gan_training'] = True/False in cofig.json.
"""


# setting up cuda and replication
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


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
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Training configs
    nz = config['training']['latent_dim']
    window_size = config['preprocessing']['window_size']
    num_epochs = config['training']['num_epochs']
    w_gan_training = config['training']['w_gan_training']
    batch_size = config['training']['batch_size']
    # W-GAN Training
    n_critic = config['training']['n_critic']
    clip_value = config['training']['clip_value']
    # Setup Adam optimizers for both G and D
    betaG = config['training']['betaG']
    betaD = config['training']['betaD']
    lrG = config['training']['lrG']  # learning rates for both
    lrD = config['training']['lrD']
    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    # Create the generator
    netG = ConvGenerator(nz, window_size)
    netG.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = ConvDiscriminator(window_size,w_gan_training)
    netD.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # data

    X_train = torch.load(config["data"]["train_path"] + f"X_train_{b_id}.pt")
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
    fixed_noise = torch.normal(0, 1, size=(16, nz, 1), device=device)
    if w_gan_training:
        # W-GAN
        print("--WGAN--")
        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            for step, data in enumerate(dataloader, 0):
                # training netD
                real_cpu = data.to(device).float()
                b_size = real_cpu.size(0)
                netD.zero_grad()

                noise = torch.randn(b_size, nz, 1, device=device)
                fake = netG(noise)
                # print(noise.shape)
                loss_D = -torch.mean(netD(real_cpu)) + torch.mean(netD(fake))
                loss_D.backward()
                optimizerD.step()

                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                if step % n_critic == 0:
                    # training netG
                    noise = torch.randn(b_size, nz, 1, device=device)
                    netG.zero_grad()
                    fake = netG(noise)
                    loss_G = -torch.mean(netD(fake))

                    netD.zero_grad()
                    netG.zero_grad()
                    loss_G.backward()
                    optimizerG.step()

                if step % 5 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, num_epochs, step, len(dataloader), loss_D.item(), loss_G.item()))
                    D_losses.append(loss_D.item())
                    G_losses.append(loss_G.item())

            # save training process
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                f, a = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(4):
                    for j in range(4):
                        a[i][j].plot(fake[i * 4 + j].view(-1))
                        a[i][j].set_xticks(())
                        a[i][j].set_yticks(())
                plt.savefig(f'plots/gen_model/{b_id}_{w_gan_training}_wgan_epoch_%d.png' % epoch)
                plt.close()
    else:
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
                noise = torch.randn(b_size, nz, 1, device=device)
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
            # save training progress ...
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                f, a = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(4):
                    for j in range(4):
                        a[i][j].plot(fake[i * 4 + j].view(-1))
                        a[i][j].set_xticks(())
                        a[i][j].set_yticks(())
                plt.savefig(f'plots/gen_model/{w_gan_training}_wgan_epoch_%d.png' % epoch)
                plt.close()



    # training plot
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # Saving the plot to a file
    plt.savefig(f'trained_out/loss_plot_{b_id}_{w_gan_training}.png')

    # save trained model
    torch.save(netD, f'trained_out/wgan_netD_{b_id}_{w_gan_training}.pth')
    torch.save(netG, f'trained_out/wgan_netG_{b_id}_{w_gan_training}.pth')

