import pandas as pd
import subprocess
import json


"""
MIT License

Copyright (c) 2021 Arun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



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
