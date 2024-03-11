import pandas as pd
import torch
import numpy as np
from preprocessing import split_sequence
import pickle
import json
from torch.autograd import Variable
import torch.nn.init as init

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
generator = torch.load(f'gan_netG_{b_id}.pth')
discriminator = torch.load(f'gan_netD_{b_id}.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
test_df = pd.read_csv(f"test_df_{b_id}.csv")
train_df = pd.read_csv(f"train_df_{b_id}.csv")


def Anomaly_score(x, G_z, Lambda=0.1):
    residual_loss = torch.sum(torch.abs(x - G_z))  # Residual Loss

    # x_feature is a rich intermediate feature representation for real data x
    output, x_feature = discriminator(x.to(device))
    # G_z_feature is a rich intermediate feature representation for fake data G(z)
    output, G_z_feature = discriminator(G_z.to(device))

    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))  # Discrimination loss

    total_loss = (1 - Lambda) * residual_loss.to(device) + Lambda * discrimination_loss
    return total_loss


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
    Anom_X = split_sequence(id_df["anomaly"], window_size)
    Isanom = Anom_X.sum(axis=1)
    id_out["labels"] = Isanom
    # reconstruct & errors
    loss_list = []
    X_ = []
    Z = []
    for i, x in enumerate(X):
        x = torch.tensor(x,device=device).view(1,window_size,1).float()
        print(x.shape)
        z = Variable(init.normal(torch.zeros(1,
                                             window_size,
                                             1), mean=0, std=0.1), requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=1e-2)

        loss = None
        for j in range(iters):  # set your interation range
            x_, _ = generator(z.cuda())
            loss = Anomaly_score(Variable(x).cuda(), x_)
            loss.backward()
            z_optimizer.step()
            X_.append(x_)
            Z.append(z)
            loss_list.append(loss)
        # optimized z, now get corresponding loss, X' and store

        loss_list.append(loss)  # Store the loss from the final iteration
        print('~~~~~~~~loss={} ~~~~~~~~~~'.format(loss))
        #

    id_out["Z"] = Z
    id_out["X_"] = X_
    id_out["recon_loss"] = loss_list

    test_out[id] = id_out

# Store the dict as pickle
with open(f'iters_{iters}_reconstruction_{b_id}.pkl', 'wb') as file:
    pickle.dump(test_out, file)
