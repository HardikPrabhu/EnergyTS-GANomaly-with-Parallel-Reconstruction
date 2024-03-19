import torch
import json
import torch.nn as nn
import random
from model import Autoencoder
import torch.optim as optim

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
    with open('../../config.json', 'r') as file:
        config = json.load(file)

    prefix = "../../"

    # Training configs
    nz = config['training']['latent_dim']
    window_size = config['preprocessing']['window_size']
    num_epochs = config['training']['num_epochs']
    w_gan_training = False  # False always
    batch_size = config['training']['batch_size']
    lr = 0.0002
    in_dim = 1

    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    # Create the Autoencoder
    autoencoder = Autoencoder(nz, window_size)
    autoencoder.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    autoencoder.apply(weights_init)

    # Print the model
    print(autoencoder)

    # data

    X_train = torch.load(f"X_train_{b_id}.pt")
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                             shuffle=True)

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # training:

    print("--Autoencoder--")
    print("Starting Training Loop...")
    # Setup loss function
    criterion = nn.MSELoss().to(device)
    for epoch in range(num_epochs):
        for i, x in enumerate(dataloader, 0):
            ############################
            # Update network: minimize  mse(x,A(x))
            ###########################
            autoencoder.zero_grad()
            real = x.to(device).float()
            output = autoencoder(real)
            err = criterion(output, real)
            err.backward()
            optimizer.step()

        # Print loss after every epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {err.item():.4f}')

    # save trained model
    torch.save(autoencoder, f'auto_{b_id}.pth')
