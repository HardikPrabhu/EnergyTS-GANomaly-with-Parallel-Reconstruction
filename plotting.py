import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import torch
import json
import matplotlib.pyplot as plt


"""
Note - Plotting can only be done after the results dataframe is created. Execute only after anom_detect_gan.py 
"""


with open('config.json', 'r') as file:
    config = json.load(file)


# params from config file
window_size = config['preprocessing']['window_size']
iters = config["recon"]["iters"]
use_dtw = config["recon"]["use_dtw"]
eval_mode = config["recon"]["use_eval_mode"]
lat_dim = 100
tol = 12

# open the results to get eval params
results_df = pd.read_csv(f"auto_results_latest_{tol}_{eval_mode}_{use_dtw}.csv")

b_ids = [1172, 1219, 1246, 1284, 1272, 1304, 91, 439, 693, 884, 896, 922, 926, 945, 968]

for b_id in b_ids:
    print(b_id)
    for dtw in [use_dtw]:  # also could check [True,False] if both are computed

        # Import the test files
        b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
        with open(f"test_out/iters_{iters}_reconstruction_{b_id}_{dtw}_{eval_mode}.pkl", "rb") as f:
            test_out_dict = pickle.load(f)
    # for plots --
    alpha = float(results_df[results_df["b_id"] == b_id]["alpha"])
    beta = float(results_df[results_df["b_id"] == b_id]["beta"])
    thresh = float(results_df[results_df["b_id"] == b_id]["thresh"])
    min_height = float(results_df[results_df["b_id"] == b_id]["min_height"])
    for i in b_df["s_no"].unique():
        id_df = b_df[b_df["s_no"] == i]
        id_dict = test_out_dict[i]
        id_df.reset_index(drop=True, inplace=True)
        error = id_dict["recon_loss"]
        z_norm = id_dict["Z"]
        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        z_norm = torch.norm(z_norm.view(-1, lat_dim), dim=1).detach().cpu().numpy()
        combined_score = alpha * error + beta * z_norm
        mask = combined_score > thresh
        if not id_dict["window_b_included"]:
            mask = np.pad(mask, (window_size // 2 - 1,), mode='constant', constant_values=False)
        positions = np.where(mask)[0]
        x = range(0, len(id_df))
        # Fit a KDE to the data
        if len(positions) > 1:
            kde = gaussian_kde(positions, bw_method=0.05)
            y = kde(x)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
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
        critical_scatter = plt.scatter([], [], marker='o', color='green', alpha=0.3, s=30, label="Critical Points")
        anomaly_scatter = plt.scatter([], [], marker='X', color='red', s=500, label='Actual Anomalies')

        for j, anomaly in enumerate(id_df["anomaly"]):
            if anomaly == 1:
                plt.scatter(j, normalized_array[j], marker='X', color='red', s=500)

        for p in peaks:
            if p < len(normalized_array):
                plt.scatter(p, normalized_array[p], marker='o', color='green', s=200)

        # Additional markers
        for p in positions:
            if p < len(normalized_array):
                plt.scatter(p, normalized_array[p], marker='o', color='green', alpha=0.3, s=30)
        threshold_line = plt.axhline(min_height, color='r', linestyle='--', label='min_height')

        # Adding legends
        plt.legend(handles=[kde_line, meter_reading_line, anomaly_scatter, peak_scatter, threshold_line, critical_scatter],
                   fontsize=font_size)

        # Making X and Y axes bold
        plt.xlabel('Time', fontsize=font_size)
        plt.ylabel('Scaled Reading', fontsize=font_size)

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.savefig(f'plots/anom_detect/{dtw}_{eval_mode}_{iters}_build_{b_id}_{i}.png',
                    dpi=300)  # Save each segment separately

        plt.close()