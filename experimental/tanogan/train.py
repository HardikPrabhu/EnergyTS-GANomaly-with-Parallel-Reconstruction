import torch
import json
import torch.nn as nn
import random
from recurrent_models_pyramid import LSTMGenerator, LSTMDiscriminator
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.init as init
import torch.optim as optim

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

    in_dim =1

    class ArgsTrn:
        workers = 4
        batch_size = 32
        epochs = num_epochs
        lr = 0.0002
        cuda = True
        manualSeed = 2


    opt_trn = ArgsTrn()

    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    # Create the generator
    netG = LSTMGenerator(1, 1,device=device)
    netG.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = LSTMDiscriminator(1,device=device)
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

    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)

    # training:
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0
    # for image generation (while training)
    fixed_noise = torch.normal(0, 1, size=(16, nz, 1), device=device)

    print("--GAN--")
    print("Starting Training Loop...")
    real_label = 1
    fake_label = 0
    # Setup loss function
    criterion = nn.BCELoss().to(device)
    for epoch in range(num_epochs):
        for i, x in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real data
            x = x.view(-1,window_size,1).float()
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            output, _ = netD.forward(real)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()

            # Train with fake data
            noise = Variable(init.normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1)).cuda()
            fake, _ = netG.forward(noise)
            output, _ = netD.forward(
                fake.detach())  # detach causes gradient is no longer being computed or stored to save memeory
            label.fill_(fake_label)
            errD_fake = criterion(output, label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise = Variable(init.normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1)).cuda()
            fake, _ = netG.forward(noise)
            label.fill_(real_label)
            output, _ = netD.forward(fake)
            errG = criterion(output, label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt_trn.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')



    # save trained model
    torch.save(netD, f'gan_netD_{b_id}.pth')
    torch.save(netG, f'gan_netG_{b_id}.pth')
