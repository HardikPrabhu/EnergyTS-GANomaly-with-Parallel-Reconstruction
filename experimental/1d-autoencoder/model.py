import torch
import torch.nn as nn


# Autoencoder -----

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.main = nn.Sequential(
            # First Conv1d layer mirroring the last ConvTranspose1d layer of the decoder
            nn.Conv1d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            # Second Conv1d layer
            nn.Conv1d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            # Third Conv1d layer
            nn.Conv1d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            # Final Conv1d layer to get to the latent space
            nn.Conv1d(256, latent_dim, int(window_size / (2 ** 3)), 1, 0, bias=False),
            nn.Tanh()  # Assuming the latent space also uses Tanh activation
        )

    def forward(self, x):
        x = self.main(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, int(window_size / (2 ** 3)), 1, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(Autoencoder, self).__init__()
        self.encoder = ConvEncoder(latent_dim, window_size)
        self.decoder = ConvDecoder(latent_dim, window_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    print("Autoencoder")
    nz = 100
    w = 48
    X = torch.normal(0, 1, size=(303, 1, w))
    model = Autoencoder(nz, w)
    Y = model(X)
    print(X.shape, Y.shape)
