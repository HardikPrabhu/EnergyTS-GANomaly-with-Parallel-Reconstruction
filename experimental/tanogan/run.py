import pandas as pd
import subprocess
import json







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

    with open('config.json', 'w') as file:
        json.dump(config, file)

    subprocess.run(["python", "preprocessing.py"])
    subprocess.run(["python", "train.py"])
    subprocess.run(["python", "test.py"])
    print(f"Processed {i}/{l} buildings ......")
    i = i + 1
