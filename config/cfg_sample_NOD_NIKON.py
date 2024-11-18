paths = {
    "root": "/data/radu/NOD/",
    "output_root": "/data/radu/NOD/nikon-set-a-stratified/",
    "rgb_path": "rgb/Nikon-npy/",
    "rggb_path": "rggb/Nikon/",
    "rgb_sample_path": "rgb-sample/",
    "rgb_extension": ".npy",
    "rggb_target_path": "rggb-target/",
    "rggb_extension": ".npy",
    "rgb_context_path": "rgb-context/",
    "rgb_files_path": "/home/radu.berdan/work/ReRAW-dev/config/splits/Nikon750_train.json",
}

cfg = {
    "rgb_max": 256,
    "n_bins": 11,
    "max_samples": 10000,
    "samples_per_channel": 2,
    "delta": 16,
    "dmiddle": (2, 2),
    "context_dim": (128, 128),
    "sample_size": (64, 64),
    "context_size_scale": 0.9,
    "type": "stratified",
}
