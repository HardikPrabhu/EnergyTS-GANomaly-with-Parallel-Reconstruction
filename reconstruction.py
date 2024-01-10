import torch
from soft_dtw_cuda import SoftDTW
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time

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
    # Calculation of speed up ...
    batch_sizes =  [5,10,25,50,100,250]
    netG = torch.load(f'trained_out/wgan_netG_1219.pth')
    # loss criterion
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    mse = torch.nn.MSELoss()
    # Record runtimes with iterations instead of epochs
    runtimes_p = []
    runtimes_s = []
    for k in batch_sizes:
            # number of examples in a batch
            print(k)
            X_train = torch.load("model_input/X_train_1219.pt")
            x = torch.tensor(X_train[:k], device=device)
            start_time = time.time()
            z, x_, loss = reconstruct(x, 1000, netG, mse, 100)
            end_time = time.time()
            runtimes_p.append(end_time - start_time)
            start_time = time.time()
            for i in x:
              z, i_, loss = reconstruct(i.view(1,1,48), 1000, netG, mse, 100)
            end_time = time.time()
            runtimes_s.append(end_time - start_time)

    plt.plot(batch_sizes, runtimes_s, 'go--', label='Sequential Reconstruction')
    plt.plot(batch_sizes, runtimes_p, 'ro--', label='Parallel Reconstruction')
    plt.legend()
    plt.xlabel('Batch Size')
    plt.ylabel('Time Taken')
    plt.savefig(f'plots/reconst.png',
                dpi=300)

    """
    for i in range(x.shape[0]):
        plt.figure(figsize=(35, 5))
        plt.plot(x[i].cpu().view(-1))
        plt.plot(x_[i].cpu().detach().view(-1))
        plt.savefig(f'plots/sample_reconstruct/_{i}.png', dpi=300)
        plt.close()
    """
