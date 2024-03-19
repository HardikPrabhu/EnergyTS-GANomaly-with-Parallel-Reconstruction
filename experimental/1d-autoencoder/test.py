import pandas as pd
import torch
import numpy as np
from preprocessing import split_sequence
import pickle
import json
import torch.nn as nn

prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)

# configs
nz = config['training']['latent_dim']
window_size = config['preprocessing']['window_size']
b_id = "all"
if config['data']["only_building"] is not None:
    b_id = config['data']["only_building"]


# model/data import
autoencoder= torch.load(f'auto_{b_id}.pth')
autoencoder.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
test_df = pd.read_csv(f"test_df_{b_id}.csv")
train_df = pd.read_csv(f"train_df_{b_id}.csv")


# testing
temp = test_df.groupby("s_no")  # group by segments
n_segs = temp.ngroups  # total number of segments
# get the segment no.s present in file
test_seg_ids = test_df["s_no"].unique()
train_seg_ids = train_df["s_no"].unique()
print(f"Total segments : {n_segs} ")
criterion = nn.MSELoss(reduction='none').to(device)
# storing reconstruction details ...
test_out = {}
for id, id_df in temp:
    id_out = {"X": None, "Z": None, "X_": None, "recon_loss": None, "labels": None, "window_b_included": False,
              "window_a_included": False}
    id_df.reset_index(drop=True, inplace=True)
    segment = np.array(id_df["meter_reading"])
    # add window length from segment before in front (if available)
    before_id = id - 1
    if before_id in test_seg_ids:
        b = test_df[test_df["s_no"] == before_id]["meter_reading"][-window_size//2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    if before_id in train_seg_ids:
        b = train_df[train_df["s_no"] == before_id]["meter_reading"][-window_size//2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    # add window length from segment after at back (if available)
    after_id = id + 1
    if after_id in test_seg_ids:
        a = test_df[test_df["s_no"] == after_id]["meter_reading"][:window_size//2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    if after_id in train_seg_ids:
        a = train_df[train_df["s_no"] == after_id]["meter_reading"][:window_size//2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    print("diff in length :", len(id_df) - len(segment), ( id_out["window_b_included"],id_out["window_a_included"]))
    # each segment will have subsequences of overlapping windows:
    X = split_sequence(segment, window_size)
    id_out["X"] = X
    # convert X to tensor and pass it through autoencoder
    X = torch.tensor(X,device=device).float().view(X.shape[0], 1, -1)
    X_ = autoencoder(X)
    loss_list = criterion(X.squeeze(),X_.squeeze()).mean(axis=1)
    # reconstruct & errors
    id_out["X_"] = X_
    id_out["recon_loss"] = np.array(loss_list.cpu().detach())
    print(loss_list.shape,X.shape)
    test_out[id] = id_out

# Store the dict as pickle
with open(f'reconstruction_{b_id}.pkl', 'wb') as file:
    pickle.dump(test_out, file)

