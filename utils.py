import matplotlib.pyplot as plt

def plot_segments(df,is_train=True):
    if is_train:
        data_type = "train"
    else:
        data_type = "test"

    for i in df["s_no"].unique():
        id_df = df[df["s_no"] == i]
        build_id = id_df["building_id"].iloc[0]
        x = range(0, len(id_df))

        plt.figure(figsize=(35, 5))

        array = id_df["meter_reading"]
        plt.plot(x, array)
        for j, anomaly in enumerate(id_df["anomaly"]):
            if anomaly == 1:
                plt.scatter(j, 0, marker='x', color='red', s=100)

        # Save the figure
        plt.savefig(f'plots/{data_type}_segments/build_{build_id}_{i}.png', dpi=300)  # Save each segment separately
        plt.close()