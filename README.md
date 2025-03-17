# ReRAW: RGB-to-RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge

<b>(27.02.2025)</b> ReRAW was accepted at CVPR2025! 🚀

This repository contains code to train ReRAW to convert RGB images into sensor specific RAW.

<div align="center">
  <img src="media/reraw-schematic.png" width="100%" /> 
</div>

## Training ReRAW
Prepare a paired dataset of RAW and RGB files, then modify the config at `./config/cfg_example.py`.

Training ReRAW involves first preparing a stratified sampling training patch dataset from the full RAW and RGB images.

### 1. Stratified Sampling
Make sure the `dataset` and `cfg_sample` parameters are set to match your dataset.
Modify `dataset` to add the original folder root, RGB images, and RAW images locations. Update the white level and black level of the sensor.
In `cfg_sample`, only the `output_folder_root` should contain the root where the resulting stratified sampling dataset will be saved.

Then run:

```bash
python3 stratified_sampling.py -c ./config/cfg_example.py -n 2
```

where `-n` denotes the number of parallel workers.

### 2. Training
With the `rgb-context`, `rgb-sample` and `rggb-target` folders are populated, run:

```bash
python3 train.py -g 0 -c ./config/cfg_example.py
```

where `-g` denotes GPU number.

The training will run and save the checkpoint at a timestamp folder in `./outputs/`.

## Running ReRAW
To run ReRAW and convert a folder of RGB images into sensor-specific RAW, run:

```bash
python3 convert.py -g 0 -f ./outputs/1731913987 -i ./example/rgb -o ./example/converted -n 2 -r False
```

where:
| Parameter | Description |
| :--- | :---- |
| -g | GPU number. |
| -f | Model output folder. |
| -i | RGB input folder. |
| -o | Output RGGB folder.  |
| -n | No of parallel workers.  |
| -r | Convert RGGB to RGB for visualisation. |



