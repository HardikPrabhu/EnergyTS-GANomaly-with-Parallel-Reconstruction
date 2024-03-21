import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd
import pickle
import torch
from bayes_opt import BayesianOptimization
import json

def get_measures(actual, pred, tol):
    """
  actual : actual labels (pointwise)
  pred :   predicted labels (pointwise)
  tol :    tolerence
  """
    TP = 0
    FN = 0
    FP = 0
    actual = np.array(actual)
    pred = np.array(pred)
    if len(pred) != 0:
        for a in actual:
            if min(abs(pred - a)) <= tol:
                TP = TP + 1
            else:
                FN = FN + 1
        for p in pred:
            if min(abs(actual - p)) > tol:
                FP = FP + 1
    else:
        FN = len(actual)

    return TP, FN, FP


# Evaluation function
def evaluate(test_df, test_out_dict,window_size, min_height, tol=24, thresh=0.5, alpha=1, beta=0,
             only_peaks=False):
    # error dict contains {s_no : [errors ]}
    TP = 0
    FN = 0
    FP = 0
    temp = test_df.groupby("s_no")
    for id, id_df in temp:
        id_df.reset_index(drop=True, inplace=True)
        id_dict = test_out_dict[id]
        error = id_dict["recon_loss"]
        z_norm = id_dict["Z"]
        error = np.array(torch.tensor(error).cpu())
        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        if beta>0:
          z_norm = torch.norm(z_norm.view(-1, lat_dim), dim=1).detach().cpu().numpy()
        else:
            z_norm = 0
        combined_score = alpha * error + beta * z_norm
        mask = combined_score > thresh
        if not id_dict["window_b_included"]:
            mask = np.pad(mask, (window_size // 2 - 1,), mode='constant', constant_values=False)

        print(len(id_df)-len(mask),id_dict["window_a_included"],id_dict["window_b_included"])

        positions = np.where(mask)[0]
        if len(positions) <= 1:
            anom = positions
        else:
            kde = gaussian_kde(positions, bw_method=0.05)
            # Evaluate the KDE at some points
            x = range(0, len(id_df))
            y = kde(x)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            if only_peaks:
                peaks, _ = find_peaks(y, height=min_height)
            else:
                peaks = np.where(y > min_height)[0]

            anom = peaks
        actual_anom = id_df.index[id_df['anomaly'] == 1]
        TP_i, FN_i, FP_i = get_measures(actual=actual_anom, pred=anom, tol=tol)
        TP = TP + TP_i
        FN = FN + FN_i
        FP = FP + FP_i
    return TP, FN, FP


if __name__ == "__main__":

    prefix = "../../"
    with open(prefix+ 'config.json', 'r') as file:
        config = json.load(file)
    # select the  iters, window size and file
    window_size = config['preprocessing']['window_size']
    iters = config["recon"]["iters"]
    lat_dim = config['training']['latent_dim']
    tolerance = [12, 24]  # the tolerance values for which evaluation is required.
    results_df = pd.DataFrame(
        columns=['b_id', 'alpha','thresh', 'min_height', 'Precision', 'Recall', 'F1','tol'])

    # get the building ids
    df = pd.read_csv(prefix + config["data"]["dataset_path"])
    b_ids = df["building_id"].unique()
    del df
    print(f"unique builds : {b_ids}")

    # b_ids = [1304] # or pass a custom list

    for b_id in b_ids:
            print(b_id)
            # Import the test files
            b_df = pd.read_csv(f"test_df_{b_id}.csv")
            with open(f"iters_{iters}_reconstruction_{b_id}.pkl", "rb") as f:
                test_out_dict = pickle.load(f)

            # optimize the params:-

            def black_box_function(thresh, min_height,alpha):
                TP, FN, FP = evaluate(b_df, test_out_dict, window_size, min_height, 6, thresh, alpha, 0)
                print(TP, FN, FP)
                try:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)
                except:
                    F1 = 0
                return F1

            # Bounded region of parameter space
            pbounds = {'thresh': (0, 100), 'min_height': (0.4, 1), 'alpha':(0,1)}

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=1, allow_duplicate_points=True
            )

            optimizer.maximize(
                init_points=70,
                n_iter=180
            )

            thresh = optimizer.max["params"]["thresh"]
            min_height = optimizer.max["params"]["min_height"]
            alpha = optimizer.max["params"]["alpha"]

            for tol in tolerance:
                TP, FN, FP = evaluate(b_df, test_out_dict, window_size, min_height, tol, thresh, alpha, 0)
                print(TP, FN, FP)
                P = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                print(P, R, F1)
                results_df.loc[len(results_df)] = [b_id, alpha, thresh, min_height, P, R, F1,tol]  # 'b_id',
                # 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1'

    results_df.to_csv(f"results.csv")
