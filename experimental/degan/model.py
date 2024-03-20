import torch
import torch.nn as nn

"""
Implementation as described in paper : https://arxiv.org/abs/2210.02449 (Degan)
"""

class ConvDiscriminator(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.linear_feat = (window_size-4)*16
        layers = [
            nn.Conv1d(1, 16, kernel_size=5, padding=0, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(in_features=self.linear_feat,out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16,out_features=1),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x, y=None):
        x = self.main(x)
        x = x.view(-1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.main = nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=window_size)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1,1,self.window_size)
        return x


if __name__ == "__main__":
    nz = 100
    w = 48
    model = Generator(nz, w)
    print(model)
    X = torch.normal(0, 1, size=(10, nz))
    Y = model(X)
    print(Y.shape)
    m = ConvDiscriminator(w)
    print(m)
    print(m(Y).shape)
    m(Y)


