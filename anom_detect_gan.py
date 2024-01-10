import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd
import pickle
import torch
from bayes_opt import BayesianOptimization
import json
#  plotting
import matplotlib.pyplot as plt


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
def evaluate(test_df, error_dict, znorm_dict, window_size, min_height, tol=24, thresh=0.5, alpha=1, beta=0,
             only_peaks=False):
    # error dict contains {s_no : [errors ]}
    TP = 0
    FN = 0
    FP = 0
    temp = test_df.groupby("s_no")
    for id, id_df in temp:
        id_df.reset_index(drop=True, inplace=True)
        error = error_dict[id]
        z_norm = znorm_dict[id]
        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        if type(z_norm) == torch.Tensor:
            z_norm = z_norm.detach().cpu().numpy()

        combined_score = alpha * error + beta * z_norm
        mask = combined_score > thresh
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

    with open('config.json', 'r') as file:
        config = json.load(file)

    # select the  iters, window size and file
    window_size = config['preprocessing']['window_size']
    iters = config["recon"]["iters"]
    use_dtw = config["recon"]["use_dtw"]

    results_df = pd.DataFrame(
        columns=['b_id', 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1'])
    for b_id in b_ids:
        print(b_id)
        for dtw in [use_dtw]:  # also could check [True,False] if both are computed

            # Import the test files
            b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
            with open(f"test_out/iters_{iters}_reconstruction_{b_id}_{dtw}.pkl", "rb") as f:
                recon_dict = pickle.load(f)

            error_dict = dict(zip(recon_dict["s_no"], recon_dict["recon_loss"]))  # error dict
            i = 0
            mse = [torch.norm(recon_dict["Z"][i].view(-1, 100), dim=1).detach().cpu().numpy() for i in
                   range(len(recon_dict["s_no"]))]
            znorm_dict = dict(zip(recon_dict["s_no"], mse))


            # optimize the two params:-

            def black_box_function(thresh, min_height, alpha, beta):
                TP, FN, FP = evaluate(b_df, error_dict, znorm_dict, window_size, min_height, 6, thresh, alpha, beta)
                print(TP, FN, FP)
                try:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)
                except:
                    F1 = 0
                return F1


            # Bounded region of parameter space
            pbounds = {'thresh': (0, 100), 'min_height': (0.4, 1), "alpha": (0, 1), "beta": (0, 1)}

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
            beta = optimizer.max["params"]["beta"]


            # for plots --
            for i in b_df["s_no"].unique():
                id_df = b_df[b_df["s_no"] == i]
                id_df.reset_index(drop=True, inplace=True)
                error = error_dict[i]
                z_norm = znorm_dict[i]
                if type(error) == torch.Tensor:
                    error = error.detach().cpu().numpy()
                if type(z_norm) == torch.Tensor:
                    z_norm = z_norm.detach().cpu().numpy()

                combined_score = alpha * error + beta * z_norm
                mask = combined_score > thresh
                num_false_values = len(id_df) - len(mask)
                mask = np.pad(mask, (0, num_false_values), mode='constant', constant_values=False)
                mask = shift_mask(mask, window_size // 2)
                positions = np.where(mask)[0]
                x = range(0, len(id_df))
                # Fit a KDE to the data
                if len(positions) > 1:
                    kde = gaussian_kde(positions, bw_method=0.05)
                    y = kde(x)
                    y = (y - np.min(y)) / (np.max(y) - np.min(y))
                    # peaks, _ = find_peaks(y, height=min_height)
                    peaks = np.where(y > min_height)[0]
                    print(peaks)
                else:
                    peaks = []
                    y = np.zeros(len(id_df))
                    if len(positions) == 1:
                        y[positions[0]] = 1
                        peaks = positions
                font_size = 18
                plt.figure(figsize=(30, 8))
                kde_line, = plt.plot(x, y, label='KDE')
                array = id_df["meter_reading"]
                normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
                meter_reading_line, = plt.plot(x, normalized_array, label='Scaled Meter Reading')

                # label handles
                peak_scatter = plt.scatter([], [], marker='o', color='green', s=200,
                                           label='Detected Anomalies (KDE above min_height)')
                critical_scatter = plt.scatter([], [], marker='o', color='blue', s=30, label="Critical Points")
                anomaly_scatter = plt.scatter([], [], marker='X', color='red', s=500, label='Actual Anomalies')

                for j, anomaly in enumerate(id_df["anomaly"]):
                    if anomaly == 1:
                        plt.scatter(j, normalized_array[j], marker='X', color='red', s=500)

                for p in peaks:
                    plt.scatter(p, normalized_array[p], marker='o', color='green', s=200)



                # Additional markers
                for p in positions:
                    plt.scatter(p, normalized_array[p], marker='o', color='blue', s=30)
                threshold_line = plt.axhline(min_height, color='r', linestyle='--', label='min_height')

                # Adding legends
                plt.legend(handles=[kde_line, meter_reading_line, anomaly_scatter, peak_scatter, threshold_line, critical_scatter],fontsize=font_size)

                # Making X and Y axes bold
                plt.xlabel('Time',fontsize=font_size)
                plt.ylabel('Scaled Reading',fontsize=font_size)

                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)

                plt.savefig(f'plots/anom_detect/{dtw}_{iters}_build_{b_id}_{i}.png',
                            dpi=300)  # Save each segment separately


                plt.close()


            TP, FN, FP = evaluate(b_df, error_dict, znorm_dict, window_size, min_height, 12, thresh, alpha, beta)
            print(TP, FN, FP)
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            print(P, R, F1)
            results_df.loc[len(results_df)] = [b_id, dtw, alpha, beta, thresh, min_height, P, R, F1]  # 'b_id',
            # 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1'

        results_df.to_csv("auto_results_latest24.csv")
