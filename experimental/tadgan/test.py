import pandas as pd
import torch
import numpy as np
from preprocessing import split_sequence
import pickle
import json
from torch.autograd import Variable
import torch.nn.init as init
from scipy import stats

def dtw_reconstruction_error(x, x_):
    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - x_[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]




prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)

# configs
nz = config['training']['latent_dim']
window_size = config['preprocessing']['window_size']
iters = config["recon"]["iters"]
b_id = "all"
if config['data']["only_building"] is not None:
    b_id = config['data']["only_building"]

# model/data import
encoder_path = f'encoder_{b_id}.pth'
decoder_path = f'decoder_{b_id}.pth'
critic_x_path = f'critic_x_{b_id}.pth'
critic_z_path = f'critic_z_{b_id}.pth'
encoder =  torch.load(encoder_path)
decoder =  torch.load(decoder_path)
critic_x = torch.load(critic_x_path)
critic_z = torch.load(critic_z_path)

test_df = pd.read_csv(f"test_df_{b_id}.csv")
train_df = pd.read_csv(f"train_df_{b_id}.csv")

# testing
temp = test_df.groupby("s_no")  # group by segments
n_segs = temp.ngroups  # total number of segments
# get the segment no.s present in file
test_seg_ids = test_df["s_no"].unique()
train_seg_ids = train_df["s_no"].unique()
print(f"Total segments : {n_segs} ")

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
    # reconstruct & errors
    loss_list = []
    X = torch.tensor(X)
    reconstructed_signal = decoder(encoder(X))
    reconstructed_signal = torch.squeeze(reconstructed_signal)
    critic_loss_list = torch.squeeze(critic_x(X)).detach().numpy()
    # X_ is directly obtained
    X_ = reconstructed_signal
    for i, x in enumerate(X):
        x_ = X_[i].detach().numpy()
        # calculate dtw-loss*critic_score as final recon loss which will be considered for anom detect.
        loss = dtw_reconstruction_error(x, x_)
        loss_list.append(loss)
        print('~~~~~~~~loss={} ~~~~~~~~~~'.format(loss))
    loss_list = stats.zscore(loss_list)
    critic_loss_list = stats.zscore(critic_loss_list)
    id_out["recon_loss"] = loss_list*critic_loss_list
    test_out[id] = id_out

# Store the dict as pickle
with open(f'iters_{iters}_reconstruction_{b_id}.pkl', 'wb') as file:
    pickle.dump(test_out, file)
