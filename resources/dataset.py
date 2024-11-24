import torchvision.transforms as transforms
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import os

class DatasetSamples(Dataset):
    def __init__(self, dataset, cfg_sample, cfg_train,  mode='train'):

        self.dataset = dataset
        self.rgb_max = 2**8
        self.transform = transforms.ToTensor()
        self.samples = []
        last_n_for_test = cfg_train["last_n_for_test"]
        self.gammas = cfg_train["gammas"]

        self.sample_path = os.path.join(cfg_sample["output_folder_root"], cfg_sample["sample_path"])
        self.context_path = os.path.join(cfg_sample["output_folder_root"], cfg_sample["context_path"])
        self.target_path = os.path.join(cfg_sample["output_folder_root"], cfg_sample["target_path"])

        self.samples = [f for f in listdir(self.sample_path) if isfile(join(self.sample_path, f))]

        if cfg_train["max_samples"] != -1:
            self.samples = self.samples[:cfg_train["max_samples"]]

        if mode == "train":
            self.samples = self.samples[:-last_n_for_test]
        else:
            self.samples = self.samples[-last_n_for_test:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = self.samples[idx]

        with open(os.path.join(self.sample_path, file_name), "rb") as f:
            sample = np.load(f, allow_pickle=True)
            sample = np.array(sample, dtype=np.float32) / self.rgb_max
        with open(os.path.join(self.context_path, file_name), "rb") as f:
            context = np.load(f, allow_pickle=True)
            context = np.array(context, dtype=np.float32) / self.rgb_max
        with open(os.path.join(self.target_path, file_name), "rb") as f:
            target = np.load(f, allow_pickle=True)
            target = (
                np.clip(np.array(target, dtype=np.float32) - self.dataset["black_level"], 0, None)
                / self.dataset["rggb_max"]
            )

        sample = self.transform(sample)
        context = self.transform(context)
        target = self.transform(target)

        targets = []
        
        for gamma in self.gammas:
            targets.append(torch.pow(target, gamma))

        targets = torch.cat(targets, dim=0)

        return sample, targets, context, target
