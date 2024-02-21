# Energy-Time-series-anomaly-detection

![True_False_1000_build_884_20.png](True_False_1000_build_884_20.png)
## Steps
1. Set up the appropriate configuration in config.json
2. Run - run.py (It runs 3 scripts and create reconstruction data pickle files)
3. Run - anom_detect_gan.py  (It also has bayes opt to tune the params)
4. Run - plotting.py to create plots for the anomaly detection

## Configs

Give below is the config file with default values.
```yaml
{"data": {"dataset_path": "dataset/15_builds_dataset.csv", "train_path": "model_input/", "only_building": 1304}, "training": {"batch_size": 128, "num_epochs": 200, "latent_dim": 100, "w_gan_training": true, "n_critic": 5, "clip_value": 0.01, "betaG": 0.5, "betaD": 0.5, "lrG": 0.0002, "lrD": 0.0002}, "preprocessing": {"normalize": true, "plot_segments": true, "store_segments": true, "window_size": 48}, "recon": {"use_dtw": true, "iters": 1000, "use_eval_mode": true}}
```

## Comments

May need to create some additional folders for more plots. (Sorry about this, will fix it!)
