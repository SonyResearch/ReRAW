dataset = {
    "name": "example-dataset",
    "root": "./example",
    "rgb_path": "rgb",
    "raw_path": "raw",
    "rggb_max": 2**14,
    "black_level": 0,
}

cfg_sample = {
    "output_folder_root": "./example",
    "sample_path": "rgb-sample/",
    "context_path": "rgb-context/",
    "target_path": "rggb-target/",
    "sample_size": (64,64),
    "delta": 64,
    "n_bins": 10,
    "type": "stratified",
    "samples_per_channel": 2,
    "context_size": (128, 128),
    "context_size_scale": 0.9
    
}

cfg_train = {
    "tag": "reraw",
    "epochs": 4,
    "batch_size": 32,
    "batch_size_test": 128,
    "lr": 1e-3,
    "lr_scaling": 0.01,
    "restart": 16,
    "hidden_size": 128,
    "depth": 8,
    "target_size": (32, 32),
    "max_samples": -1,
    "save_path": "./outputs/",
    "tensorboard_path": "./runs/run-1/",
    "last_n_for_test": 5,
    "gammas": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
}