import pandas as pd
import numpy as np
import json
import torch
import os


def impute_nulls(data):
    """
    imputation: mean for all the buildings is used for imputing (This could be improved later).
    """
    mean_reading = data.groupby('building_id').mean()['meter_reading']
    building_id = mean_reading.index
    values = mean_reading.values

    for i, idx in enumerate(building_id):
        data[data['building_id'] == idx] = data[data['building_id'] == idx].fillna(values[i])
    return data


def preprocess_data(data, dropna=True):
    """
    Preprocessing: for now dropping missing value is used.
    Users could add more preprocessing steps.
    """
    data = data.sort_values(by="timestamp")
    print(f'unique buildings : {data["building_id"].unique()}')
    if dropna:
        data.dropna(subset=['meter_reading'], inplace=True)
    else:
        data = impute_nulls(data)
    return data


def segment_data(data, n_segments=25, normalize=True):
    """
    A function which will divide the datasets in anomlous and normal segments. (train/test)
    parameters:
    ---------
    data: pandas dataframe
    Dataset with columns "building_id","anomaly" and "meter reading"

    n_segments: int
    Number of segments in which the timeseries data for each building  should be divided.

    normalize: bool
    If true, each segment is normalize to lie in the range [-1,1]
    """
    min_seg_len = len(data)
    temp = data.groupby("building_id")
    normal_df = pd.DataFrame()  # train, valid (all normal entries)
    ano_df = pd.DataFrame()  # test (mixed)
    s_no = 0
    for id, id_df in temp:  # The column remains in id_df
        segments_per_build = np.array_split(id_df, n_segments)
        for d in segments_per_build:
            if len(d) < min_seg_len:
                min_seg_len = len(d)
            s_no = s_no + 1
            d["s_no"] = s_no
            if normalize == True:
                seq_x = d["meter_reading"]
                if np.max(seq_x) - np.min(seq_x) == 0:
                    seq_x = np.zeros(seq_x.shape)
                else:
                    seq_x = 2 * (seq_x - np.min(seq_x)) / (np.max(seq_x) - np.min(seq_x)) - 1
                d["meter_reading"] = seq_x

            if 1 in d['anomaly'].values:
                ano_df = pd.concat([ano_df, d])
            else:
                normal_df = pd.concat([normal_df, d])

    del temp

    return normal_df, ano_df, s_no, min_seg_len  # s_no to get the total number of segments


def split_sequence(sequence, n_steps, centering=False, minmax=False):
    """A function which will create the inputs for the models"""
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_x = np.array(seq_x)
        if centering == True:
            std = np.std(seq_x)
            if std == 0:
                seq_x = np.zeros(seq_x.shape)
            else:
                seq_x = (seq_x - np.mean(seq_x)) / std
        if minmax == True:
            if np.max(seq_x) - np.min(seq_x) == 0:
                seq_x = np.zeros(seq_x.shape)
            else:
                seq_x = 2 * (seq_x - np.min(seq_x)) / (np.max(seq_x) - np.min(seq_x)) - 1

        X.append(seq_x)
    X = np.array(X)
    return X


if __name__ == "__main__":
    prefix = "../../"
    # Load config
    with open(prefix + 'config.json', 'r') as file:
        config = json.load(file)

    window_size = config['preprocessing']['window_size']
    b_id = "all"

    # train-test segments
    data = pd.read_csv(prefix+f"{config['data']['dataset_path']}")
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]   # one particular building at a time (recommended)
        data = data[data["building_id"] == b_id]

    data = preprocess_data(data)
    train_df, test_df, s_no, min_len = segment_data(data,normalize=False)
    print(f"total number of segments : {s_no}")
    print(f" min len of segment: {min_len}")


    # storing segments
    train_df.to_csv(f"train_df_{b_id}.csv", index=False)
    test_df.to_csv(f"test_df_{b_id}.csv", index=False)  # will be used for testing later

    # Convert training data into model input:
    X_train = []
    seg_count = 0
    temp = train_df.groupby("s_no")
    for id, id_df in temp:
        X_w = split_sequence(id_df["meter_reading"], window_size)
        X_train.extend(X_w)
        seg_count += 1
    X_train = np.array(X_train)
    X_train = X_train.reshape(len(X_train), 1, -1)

    print(f"training tensor shape: {X_train.shape}")
    torch.save(X_train, f"X_train_{b_id}.pt" )
    print(f'The model training input is stored at : {os.getcwd()}')
