# Energy-Time-series-anomaly-detection

Our related paper "Generative Adversarial Network with Soft-Dynamic Time Warping and Parallel Reconstruction for Energy Time Series Anomaly Detection" (https://doi.org/10.48550/arXiv.2402.14384) got accepted at the AI4TS Workshop @ AAAI 24. 

**Note :** A journal extension of the paper is under development. The repository will be frequently updated.

![True_True_1000_build_977_10.png](True_True_1000_build_977_10.png)

[View Poster PDF](poster.pdf)

## LEAD Dataset

This study uses the LEAD 1.0 dataset. This public dataset includes hourly-based electricity meter readings for commercial buildings over up to one year. The dataset consists of year-long hourly electricity meter readings with anomaly annotations from 200 buildings and is open-sourced in this GitHub repository [6]. Each building contains about 8,747 data points. Anomaly annotations are provided, marking anomalous points (timestamps) within each building's time series.

This dataset contains hourly electricity meter readings and anomaly annotations for various commercial buildings over a period of up to one year. The data is structured as follows:

- `building_id`: Unique identifier for each building.
- `timestamp`: Hourly timestamp of the meter reading.
- `meter_reading`: Actual electricity meter reading value.
- `anomaly`: Binary indicator of whether the timestamp (reading) is considered anomalous (1) or not (0).

The dataset covers readings from 200 buildings, with each building having approximately 8,747 data points. Anomaly annotations are provided to mark specific timestamps within each building's time series where anomalous readings were detected.

Here's a small example of the dataset:

| building_id | timestamp       | meter_reading | anomaly |
|-------------|-----------------|---------------|---------|
| 1           | 01-01-2016 00:00| 100.5         | 0       |
| 1           | 01-01-2016 01:00| 98.2          | 0       |
| 1           | 01-01-2016 02:00| 95.7          | 0       |
| 2           | 01-01-2016 00:00| 200.1         | 0       |
| 2           | 01-01-2016 01:00| 203.4         | 1       |
| 2           | 01-01-2016 02:00| 197.8         | 0       |


## Getting started 

1. Clone the repository:
  ```bash
  git clone https://github.com/HardikPrabhu/Energy-Time-Series-Anomaly-Detection.git
  ```
2. Navigate to the cloned repository.
3. Install the required python packages using pip:

  ```bash
  pip install -r requirements.txt
  ```

4. Adjust the experiment settings:

  Modify the config.json file to configure the experiment according to your requirements. This [JSON config file](config.json) allows you to customize various parameters and settings for your experiments.
  Given below is the config file with default values.

```yaml
{
    "data": {
        "dataset_path": "dataset/15_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": true,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": true,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```


## Steps
1. Set up the appropriate configuration in config.json
2. Run - run.py (It runs 3 scripts and create reconstruction data pickle files)
3. Run - anom_detect_gan.py  (It also has bayes opt to tune the params)
4. Run - plotting.py to create plots for the anomaly detection


## Other Methodologies


The directory "experimental" contains code for comparisons with other popular Gan based methods. We perform anomaly detection using different methodologies and also try to maintain similar evaluation and training hyper-parameters for fair comparisons.

It includes the following implementations:

[1] TAnoGAN - Use gradient descent in the noise space, to get appropriate noise for reconstruction.

[2] TADGAN - Train an encoder with cycle consistency, in order to map back to the noise space for reconstruction. 

[3] 1-D CNN Autoencoder - Use the reconstruction error obtained by encoding-decoding as anomaly score.

[4] DEGAN - Use the output of the discriminator (1- D(x)) directly as a score. 



## Useful Resources

[1] 1D-DCGAN : https://github.com/LixiangHan/GANs-for-1D-Signal

[2] soft-dtw loss cuda : https://github.com/Maghoumi/pytorch-softdtw-cuda

[3] TAnoGAN : https://github.com/mdabashar/TAnoGAN

[4] MADGAN : https://github.com/Guillem96/madgan-pytorch

[5] TADGAN : https://github.com/arunppsg/TadGAN

[6] LEAD Dataset : https://github.com/samy101/lead-dataset

[7] DEGAN : https://arxiv.org/pdf/2210.02449.pdf

