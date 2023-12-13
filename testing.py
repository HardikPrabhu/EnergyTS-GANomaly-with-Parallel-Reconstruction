import pandas as pd
import torch
from preprocessing import split_sequence
from soft_dtw_cuda import SoftDTW
from reconstruction import reconstruct
import pickle
import json

with open('config.json', 'r') as file:
    config = json.load(file)

# configs

eval_mode = False
nz = config['training']['latent_dim']
window_size = config['preprocessing']['window_size']
b_id = "all"
if config['data']["only_building"] is not None:
    b_id = config['data']["only_building"]

if config["training"]["w_gan_training"]:
    train_type = "wgan"
    # model
    netG = torch.load(f'trained_out/{train_type}_netG_{b_id}.pth')
else:
    train_type = "autoencoder"
    autoencoder = torch.load(f'trained_out/autoencoder_{b_id}.pth')

iters = config["recon"]["iters"]
use_dtw = config["recon"]["use_dtw"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")

temp = test_df.groupby("s_no")  # group by segments
n_segs = temp.ngroups  # total number of segments
error_gen = {}  # dict to store reconstruction information
i = 1  # counter
# storing reconstruction details ...
test_out = {"s_no": [], "X": [], "Z": [], "X_": [], "recon_loss": [], "labels": []}
for id, id_df in temp:
    print(f"{i}/{n_segs}")
    id_df.reset_index(drop=True, inplace=True)
    test_out["s_no"].append(id)
    segment = id_df["meter_reading"]
    print(len(segment))
    # each segment will have subsequences of overlapping windows:
    X = split_sequence(id_df["meter_reading"], window_size)
    test_out["X"].append(X)
    Anom_X = split_sequence(id_df["anomaly"], window_size)
    Isanom = Anom_X.sum(axis=1)
    test_out["labels"].append(Isanom)
    # reconstruct & errors
    if use_dtw:
        criterion = SoftDTW(use_cuda=True, gamma=0.1)
    else:
        criterion = torch.nn.MSELoss()

    if train_type == "wgan":
        print(X.shape)
        if eval_mode:
            netG.eval()  # batch norm in static mode
        X = torch.tensor(X, device=device).view(X.shape[0], 1, -1)
        Z, X_, loss = reconstruct(X, iters, netG, criterion, 100)
        test_out["Z"].append(Z)
        test_out["X_"].append(X_)
        test_out["recon_loss"].append(loss)


    else:
        autoencoder.eval()
        print(X.shape)
        X = torch.tensor(X, device=device).view(X.shape[0], 1, -1).float()
        Y = autoencoder(X)
        X = X.view(-1, 48)
        Y = Y.view(-1, 48)
        print(X.shape, Y.shape)

        loss = ((X - Y) ** 2).mean(1)

        test_out["X_"].append(Y)
        test_out["recon_loss"].append(loss)

    i = i + 1

if train_type == "wgan":
    # Store the dict as pickle
    with open(f'test_out/iters_{iters}_reconstruction_{b_id}_{use_dtw}.pkl', 'wb') as file:
        pickle.dump(test_out, file)

else:
    with open(f'test_out/autoencoder_reconstruction_{b_id}.pkl', 'wb') as file:
        pickle.dump(test_out, file)
