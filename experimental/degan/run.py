import pandas as pd
import subprocess
import json
import time
import sys

# To perform all the operations (-one model per building)
prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)

# open and get all the unique buildings:
df = pd.read_csv(prefix + config["data"]["dataset_path"])
uni_b = df["building_id"].unique()
print(f"unique builds : {uni_b}")
l = len(uni_b)
i = 1  # counter for builds

for b_id in df["building_id"].unique():
    print(b_id)
    # Change configs
    config["data"]["only_building"] = int(b_id)

    with open(prefix + 'config.json', 'w') as file:
        json.dump(config, file)

    start_time = time.time()
    subprocess.run([sys.executable, "preprocessing.py"])
    end_time = time.time()
    pre_time = end_time - start_time
    print(f"Building {b_id} :: Time taken for preprocessing : {pre_time}")

    starting_time = time.time()
    subprocess.run([sys.executable, "train.py"])
    end_time = time.time()
    train_time = end_time - starting_time
    print(f"Building {b_id} :: Time taken for training : {train_time}")

    start_time = time.time()
    subprocess.run([sys.executable, "test.py"])
    end_time = time.time()
    test_time = end_time - start_time
    print(f"Building {b_id} :: Time taken for test : {test_time}")

    print(f"Processed {i}/{l} buildings ...... Total time taken: {end_time - starting_time} secs")

    i = i + 1
