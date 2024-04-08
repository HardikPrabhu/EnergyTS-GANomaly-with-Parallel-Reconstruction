import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

custom_params = {"axes.spines.right": True, "axes.spines.top": True}


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
results_df = pd.read_csv(f"results_eval_mode_on_soft_dtw.csv")

b_ids = [1272, 977,1219,889,884, 886, 147,931] # select buildings to plot ...

for b_id in b_ids:
    print(b_id)
    for dtw in [use_dtw]:  # also could check [True,False] if both are computed

        # Import the test files
        b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
        with open(f"test_out/iters_{iters}_reconstruction_{b_id}_{dtw}_{eval_mode}.pkl", "rb") as f:
            test_out_dict = pickle.load(f)
    # for plots --
    alpha = float(results_df[results_df["b_id"] == b_id]["alpha"].iloc[0])
    beta = float(results_df[results_df["b_id"] == b_id]["beta"].iloc[0])
    thresh = float(results_df[results_df["b_id"] == b_id]["thresh"].iloc[0])
    min_height = float(results_df[results_df["b_id"] == b_id]["min_height"].iloc[0])
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


        font_size = 20
        plt.figure(figsize=(30, 8))
        sns.set_style("ticks")

        # Set the Seaborn style and color palette

        kde_line, = plt.plot(x, y, label='KDE', linewidth=2)

        array = id_df["meter_reading"]
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))


        # label handles
        critical_scatter = plt.scatter([], [], marker='o', color='green', alpha=0.85, s=50, label="Critical Points")


        apeaks = np.where(id_df["anomaly"]>0)[0]

        # Draw shaded region for actual anomalies
        start_positions = []
        end_positions = []
        for i in range(len(apeaks)):
            if i == 0 or apeaks[i] - apeaks[i - 1] > 1:
                start_positions.append(apeaks[i])
            if i == len(apeaks) - 1 or apeaks[i + 1] - apeaks[i] > 1:
                end_positions.append(apeaks[i])

        # Draw shaded rectangles for consecutive anomalies
        for start, end in zip(start_positions, end_positions):
            if end - start > 0:
                plt.axvspan(start - 0.5, end + 0.5, alpha=0.2, color='red')
            else:
                plt.axvspan(start - 0.5, start + 0.5, alpha=0.2, color='red')


        meter_reading_line, = plt.plot(x, normalized_array, label='Scaled Meter Reading', linewidth=3.5)

        # Draw shaded rectangles for predicted anomalous regions
        start_positions = []
        end_positions = []
        for i in range(len(peaks)):
            if i == 0 or peaks[i] - peaks[i - 1] > 1:
                start_positions.append(peaks[i])
            if i == len(peaks) - 1 or peaks[i + 1] - peaks[i] > 1:
                end_positions.append(peaks[i])

        # Draw shaded rectangles for consecutive anomalies
        for start, end in zip(start_positions, end_positions):
            if end - start > 0:
                plt.axvspan(start - 0.5, end + 0.5, alpha=0.12, color='green')
            else:
                plt.axvspan(start - 0.5, start + 0.5, alpha=0.12, color='green')

        # Plot critical points
        for p in positions:
            if p < len(normalized_array):
                plt.scatter(p, 0, marker='o', color='green', alpha=0.85, s=50)

        threshold_line = plt.axhline(min_height, color='r', linestyle='--', label='min_height', linewidth=2)

        shaded_patch = mpatches.Patch(color='green', alpha=0.12, label='Predicted Anomalies')
        shaded_patch_r = mpatches.Patch(color='red', alpha=0.2, label='Actual Anomalies')
        # Adding legends
        plt.legend(handles=[kde_line, meter_reading_line, threshold_line, critical_scatter,shaded_patch,shaded_patch_r],
                   fontsize=font_size)

        plt.xlabel('Timestamp', fontsize=font_size)
        plt.ylabel('Scaled Reading', fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.xticks(fontsize=font_size)
        num_ticks = 7  # Number of timestamps to display
        tick_indices = np.linspace(0, len(id_df) - 1, num_ticks, dtype=int)
        tick_labels = id_df["timestamp"][tick_indices]
        plt.xticks(ticks=tick_indices, labels=tick_labels)

        plt.savefig(f'plots/anom_detect/{dtw}_{eval_mode}_{iters}_build_{b_id}_{i}.png', dpi=300)
        plt.close()