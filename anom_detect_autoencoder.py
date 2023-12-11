import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd
import pickle
import torch
from bayes_opt import BayesianOptimization

def shift_mask(mask, k):
    # Split the mask into two parts
    mask1 = mask[:-k]
    # Shift the parts separately
    shifted_mask = np.concatenate((np.zeros(k, dtype=bool), mask1))
    return shifted_mask


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
def evaluate(test_df, error_dict, window_size, min_height, tol=24, thresh=0.5,
             only_peaks=False):
    # error dict contains {s_no : [errors ]}
    TP = 0
    FN = 0
    FP = 0
    temp = test_df.groupby("s_no")
    for id, id_df in temp:
        id_df.reset_index(drop=True, inplace=True)
        error = error_dict[id]

        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        print(error.shape,len(id_df))
        mask = error > thresh
        num_false_values = len(id_df) - len(mask)
        mask = np.pad(mask, (0, num_false_values), mode='constant', constant_values=False)
        mask = shift_mask(mask, window_size // 2)
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


    b_ids = [1172, 1219, 1246, 1284, 1272, 1304, 91, 439, 693, 884, 896, 922, 926, 945, 968]

    results_df = pd.DataFrame(
        columns=['b_id', 'thresh', 'min_height', 'Precision', 'Recall', 'F1'])
    for b_id in b_ids:
        print(b_id)
        # select the building
        window_size = 48


        # Import the test files
        b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
        with open(f"test_out/autoencoder_reconstruction_{b_id}.pkl", "rb") as f:
            recon_dict = pickle.load(f)

        error_dict = dict(zip(recon_dict["s_no"], recon_dict["recon_loss"]))  # error dict


        # optimize the two params:-

        def black_box_function(thresh, min_height):
            TP, FN, FP = evaluate(b_df, error_dict, window_size, min_height, 6, thresh)
            print(TP, FN, FP)
            try:
                P = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
            except:
                F1 = 0
            return F1


        # Bounded region of parameter space
        pbounds = {'thresh': (0, 100), 'min_height': (0.4, 1)}

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



        TP, FN, FP = evaluate(b_df, error_dict, window_size, min_height, 24, thresh)
        print(TP, FN, FP)
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(P, R, F1)
        results_df.loc[len(results_df)] = [b_id, thresh, min_height, P, R, F1]

    results_df.to_csv("auto_results_latest24.csv")
