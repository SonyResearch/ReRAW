import sys
import torch
import numpy as np
import cv2
import argparse
import numpy as np

import os
from os import listdir
from os.path import isfile, join

from resources.models import ReRAW
from resources.utils import convert_image, convert_to_rgb
import time

import threading
import queue
import math
import json

import multiprocessing
from multiprocessing import set_start_method
from tqdm import tqdm

def run_reraw(model, file_name, input_path, output_path, cfg_sample, cfg_train, dataset, cuda_no, to_rgb):

    device = torch.device(f"cuda:{cuda_no}")  # GPU 1 is 'cuda:1'
    torch.cuda.set_device(device)

    file_name_, extension = file_name.split('.')

    input_file_path = os.path.join(input_path, file_name)

    if extension == 'npy':
        original_rgb = np.load(input_file_path)
    elif extension in ['JPG', 'png', 'jpg', 'jpeg', 'PNG']:
        original_rgb = cv2.imread(input_file_path)
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
    
    h, w = original_rgb.shape[:2]
    if h % 2 != 0:
        original_rgb = cv2.resize(original_rgb, (w, h+1), interpolation=cv2.INTER_AREA)
    if w % 2 != 0:
        original_rgb = cv2.resize(original_rgb, (w+1, h), interpolation=cv2.INTER_AREA)
    
    original_rgb = original_rgb.astype(np.float32) / 256

    converted_rggb = convert_image(
        model,
        original_rgb,
        context_size=cfg_sample["context_size"],
        sample_size=cfg_sample["sample_size"],
        gammas=cfg_train["gammas"],
        batch_size=cfg_train["batch_size_test"]
    )

    if to_rgb:
        converted_rgb = convert_to_rgb(converted_rggb, gamma=0.5)
        converted_rgb = cv2.cvtColor(converted_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, file_name_ + '.jpg'), converted_rgb)


    rggb_max = dataset['rggb_max']
    converted_rggb = np.clip(converted_rggb * rggb_max, 0, rggb_max-1).astype(np.uint16)
    save_rggb_path = os.path.join(output_path, file_name_ + '.npy')
    np.save(save_rggb_path, converted_rggb)


if __name__ == "__main__":

    set_start_method("spawn")
    parser = argparse.ArgumentParser(description="ReRAW -> converting RGBs to RAW images.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU used to convert the images.")
    parser.add_argument("-f", "--folder", type=str, default="", help="Checkpoint folder")
    parser.add_argument("-i", "--input_path", type=str, default="", help="Input folder")
    parser.add_argument("-o", "--output_path", type=str, default="", help="Output folder")
    parser.add_argument("-n", "--n", type=int, default=0, help="Number of processes.")
    parser.add_argument("-r", "--rgb", type=bool, default=False, help="convert also to rgb for visualisation.")
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    device = torch.device(f"cuda:{cuda_no}") 
    torch.cuda.set_device(device)

    checkpoint_folder = args.folder
    input_path = args.input_path
    output_path = args.output_path
    N_MODELS = int(args.n)

    os.makedirs(output_path, exist_ok=True)

    cfg_path = os.path.join(checkpoint_folder, "cfg.py")
    model_path = os.path.join(checkpoint_folder, "best.pth")

    variables = {}
    with open(cfg_path) as file:
        exec(file.read(), variables)
    cfg_sample = variables["cfg_sample"]
    cfg_train = variables["cfg_train"]
    dataset = variables["dataset"]

    input_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # DEBUG = True
    models = []
    for i in range(N_MODELS):
        model = ReRAW(
            in_size=3,
            out_size=4,
            target_size=cfg_train["target_size"],
            hidden_size=cfg_train["hidden_size"],
            n_layers=cfg_train["depth"],
            gammas=cfg_train["gammas"]
        )

        model.load_state_dict(torch.load(model_path))
        model.to(f"cuda:{cuda_no}")
        model.eval()
        models.append(model)

    tasks = []

    for i, file_name in enumerate(input_files):
        tasks.append([models[i % N_MODELS], file_name, input_path, output_path, cfg_sample, cfg_train, dataset, args.gpu, args.rgb])

        # if DEBUG:
        #     run_reraw(*tasks[-1])

    

    pool = multiprocessing.Pool(processes=N_MODELS)

    with tqdm(total=len(tasks)) as pbar:

        # Function to update progress bar
        def update(result):
            pbar.update()

        # Use apply_async in a loop to manually track progress
        for task in tasks:
            pool.apply_async(run_reraw, args=task, callback=update)

        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()

