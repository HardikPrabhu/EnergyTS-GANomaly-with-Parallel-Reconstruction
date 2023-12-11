import torch
from soft_dtw_cuda import SoftDTW
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reconstruct(x, iters, netG, criterion=None, latent_dim=10):
    loss_array = None
    x_ = None
    z = torch.zeros(size=(x.shape[0], latent_dim, 1), device=device)
    z.requires_grad = True
    optimizerz = torch.optim.Adam([z], lr=0.1)
    for i in range(iters):
        optimizerz.zero_grad()
        x_ = netG(z)
        # Compute the loss value
        loss_array = criterion(x.float(), x_)
        loss = loss_array.mean()
        loss.backward()
        optimizerz.step()

    return z, x_, loss_array


if __name__ == "__main__":
    # Loading model in eval mode:
    netG = torch.load(f'trained_out/wgan_netG_1219.pth')
    netG.eval()
    # testing mode
    X_train = torch.load("model_input/X_train_1219.pt")
    x = torch.tensor(X_train[:256][::20], device=device)

    # loss criterion
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    mse = torch.nn.MSELoss()
    z, x_, loss = reconstruct(x, 400, netG, mse, 100)
    print(loss)

    for i in range(x.shape[0]):
        plt.figure(figsize=(35, 5))
        plt.plot(x[i].cpu().view(-1))
        plt.plot(x_[i].cpu().detach().view(-1))
        plt.savefig(f'plots/sample_reconstruct/_{i}.png', dpi=300)
        plt.close()
    
