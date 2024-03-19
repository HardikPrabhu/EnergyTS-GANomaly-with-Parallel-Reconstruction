import torch
import torch.nn as nn

""" 
1-D DCGAN Model
-----------------
Architecture: Generator and Discriminator with 1-D conv layers. 3 symmetrical hidden layers. 
Parameters:
         window_size : size of the subsequence (int)
         nz : size of the latent dimension (int)
         wgan_train : Choice for Discriminator based on training (bool) 
"""


class ConvDiscriminator(nn.Module):
    def __init__(self, window_size,wgan_train=True):
        self.wgan_train = wgan_train
        super().__init__()
        layers = [
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 1, kernel_size=int(window_size / (2 ** 3)), stride=2, padding=0, bias=False)
        ]

        if not wgan_train:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x, y=None):
        x = self.main(x)
        x = x.view(-1)
        return x


class ConvGenerator(nn.Module):
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



if __name__ == "__main__":
    nz = 100
    w = 48
    model = ConvGenerator(nz, w)
    print(model)
    X = torch.normal(0, 1, size=(10, nz, 1)) # chanel dim is latent
    Y = model(X)
    print(Y.shape)
    m = ConvDiscriminator(w)
    print(m)
    print(m(Y).shape)
    m(Y)


