import torch
import json
import torch.nn as nn
import random
from model import ConvDiscriminator, ConvGenerator, Autoencoder
import matplotlib.pyplot as plt

# setting up cuda and replication
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


def weights_init(m):
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
    netD = ConvDiscriminator(window_size)
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
    if config["training"]["w_gan_training"]:
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
                plt.savefig('plots/gen_model/wgan_epoch_%d.png' % epoch)
                plt.close()

        # training plot
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()

        # Saving the plot to a file
        plt.savefig(f'trained_out/loss_plot_{b_id}.png')

        # save trained model
        torch.save(netD, f'trained_out/wgan_netD_{b_id}.pth')
        torch.save(netG, f'trained_out/wgan_netG_{b_id}.pth')

    else:

        # Autoencoder Training : for comparison
        autoencoder = Autoencoder(nz,window_size)
        if torch.cuda.is_available():
            autoencoder.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            for data in dataloader:
                # If using GPU
                if torch.cuda.is_available():
                    data = data.cuda().float()

                # Forward pass
                output = autoencoder(data)
                loss = criterion(output, data)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save(autoencoder, f'trained_out/autoencoder_{b_id}.pth')