import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import time
import os
import json
import sys
from resources.utils import large_crop, convert_to_rgb, load_raw
import multiprocessing
from tqdm import tqdm
import random
import argparse


def sample_from_image(file_name, rgb_path, raw_path, rgb_extension, raw_extension, cfg_sample):
    rgb_sample_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['sample_path'])
    rggb_target_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['target_path'])
    rgb_context_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['context_path'])

    n_samples = cfg_sample["samples_per_channel"]

    if rgb_extension == "npy":
        rgb_img = np.load(os.path.join(rgb_path, f'{file_name}.{rgb_extension}')).astype(np.float32)
    elif rgb_extension in ["jpg", "JPG", "jpeg", "png"]:
        rgb_img = cv2.imread(os.path.join(rgb_path, f'{file_name}.{rgb_extension}'))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    rggb_img = load_raw(os.path.join(raw_path, f'{file_name}.{raw_extension}'))

    img_h, img_w = rgb_img.shape[:2]

    count_h = (img_h - cfg_sample["sample_size"][0]) // cfg_sample["delta"]
    count_w = (img_w - cfg_sample["sample_size"][1]) // cfg_sample["delta"]

    bright = np.zeros((count_h, count_w, 3))

    for i in range(1, count_h - 1):
        for j in range(1, count_w - 1):
            y = cfg_sample["sample_size"][0] + i * cfg_sample["delta"]
            x = cfg_sample["sample_size"][1] + j * cfg_sample["delta"]
            for c in range(3):
                brightness = (
                    np.mean(
                        rgb_img[
                            y - cfg_sample["sample_size"][0]//2 : y + cfg_sample["sample_size"][0]//2,
                            x - cfg_sample["sample_size"][1]//2 : x + cfg_sample["sample_size"][1]//2,
                            c,
                        ]
                    )
                    / 256
                )
                bright[i, j, c] = brightness
    bins = np.linspace(0, 1, cfg_sample["n_bins"]+1)
    counter = 0

    for c in range(3):
        idx = np.digitize(bright[:, :, c], bins)

        locations = [np.where(idx == i) for i in range(1, cfg_sample["n_bins"])]

        # ###############
        prob = [int(len(loc)) > 0 for loc in locations]
        prob = [p / np.sum(prob) for p in prob]
        sampled_indexes = [x for x in range(len(prob))]
        # ###############

        for i in range(n_samples):
            if cfg_sample["type"] == "random":
                y = np.random.randint(1, count_h - 1)
                x = np.random.randint(1, count_w - 1)
            else:
                try:
                    sampled_index = np.random.randint(cfg_sample['n_bins'])
                    sampled_location = np.random.randint(len(locations[sampled_index][0]))
                    y = locations[sampled_index][0][sampled_location]
                    x = locations[sampled_index][1][sampled_location]
                except:
                    y = np.random.randint(1, count_h - 1)
                    x = np.random.randint(1, count_w - 1)

            y1 = np.clip(y * cfg_sample["delta"], 2, None)
            y2 = np.clip(y1 + cfg_sample["sample_size"][0], None, img_h - 2)

            x1 = np.clip(x * cfg_sample["delta"], 2, None)
            x2 = np.clip(x1 + cfg_sample["sample_size"][1], None, img_w - 2)

            selected_patch_rgb = rgb_img[y1 - 1 : y2 + 1, x1 - 1 : x2 + 1]
            selected_patch_rggb = np.asarray(rggb_img[y1 // 2 : y2 // 2, x1 // 2 : x2 // 2])
            context = large_crop(rgb_img, (x1, y1), cfg_sample["context_size_scale"], cfg_sample["context_size"])

            np.save(rgb_sample_path + f"{file_name}_{counter}.npy", selected_patch_rgb)
            np.save(rggb_target_path + f"{file_name}_{counter}.npy", selected_patch_rggb)
            np.save(rgb_context_path + f"{file_name}_{counter}.npy", context)
            counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform stratified sampling.")
    parser.add_argument("-c", "--cfg", type=str, default="", help="Config path.")
    parser.add_argument("-n", "--n", type=int, default=2, help="number of workers")
    args = parser.parse_args()

    cfg_path = args.cfg
    N_WORKERS = int(args.n)

    variables = {}
    with open(cfg_path) as file:
        exec(file.read(), variables)

    dataset = variables["dataset"]
    cfg_train = variables["cfg_train"]
    cfg_sample = variables["cfg_sample"]

    rgb_path = os.path.join(dataset['root'], dataset['rgb_path'])
    raw_path = os.path.join(dataset['root'], dataset['raw_path'])

    rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
    raw_files = [f for f in os.listdir(raw_path) if os.path.isfile(os.path.join(raw_path, f))]
    rgb_extension = rgb_files[0].split('.')[-1]
    raw_extension = raw_files[0].split('.')[-1]

    rgb_sample_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['sample_path'])
    rggb_target_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['target_path'])
    rgb_context_path = os.path.join(cfg_sample['output_folder_root'], cfg_sample['context_path'])

    for folder in [rgb_sample_path, rggb_target_path, rgb_context_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    file_names = [file_name.split('.')[0] for file_name in rgb_files]

    tasks = [
        [file_names[i], rgb_path, raw_path, rgb_extension, raw_extension, cfg_sample]
        for i in range(len(file_names))
    ]

    # for task in tasks:
    #     sample_from_image(*task)

    # exit()

    pool = multiprocessing.Pool(processes=N_WORKERS)
    with tqdm(total=len(tasks)) as pbar:
        results = []

        def update(result):
            pbar.update()

        for args in tasks:
            pool.apply_async(sample_from_image, args=args, callback=update)

        pool.close()
        pool.join()
