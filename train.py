import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import time
import cv2
from torch.utils.data import DataLoader
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from resources.models import ReRAW, hard_log_loss
from resources.dataset import DatasetSamples
from resources.utils import (
    convert_image,
    save_model,
    save_cfg,
    test_patches,
    load_cfg,
)

import sys
import argparse
import multiprocessing
from multiprocessing import set_start_method


def train(model, dataloader, testloader, save_path, cfg, writer):
    lr = cfg["lr"]
    lr_scaling = cfg["lr_scaling"]
    restart = cfg["restart"]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, restart, eta_min=lr * lr_scaling
    )

    criterion1 = hard_log_loss()
    criterion2 = hard_log_loss()
    psnr_max = 0

    for epoch in range(cfg["epochs"]):
        model.train()

        with tqdm(dataloader, unit="batch", disable=False) as tepoch:
            loss = torch.tensor(0)
            last_loss = 0
            for data in tepoch:
                last_loss = last_loss * 0.9 + loss.item() * 0.1
                tepoch.set_description(
                    f'------->>>> Epoch {epoch} | loss={"%.5f" % round(last_loss, 5)}'
                )
                sample, targets, context, target = (
                    data[0].cuda(),
                    data[1].cuda(),
                    data[2].cuda(),
                    data[3].cuda(),
                )

                optimizer.zero_grad()
                y, outputs, _ = model(sample, context)
                loss1 = criterion1(outputs, targets)
                loss2 = criterion2(y, target)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

        writer.add_scalar(f"loss", loss.item(), epoch)

        psnr = test_patches(model, testloader, gammas=cfg["gammas"])

        writer.add_scalar("psnr_mean_patch", psnr, epoch)
        if psnr > psnr_max:
            save_model(model, os.path.join(save_path, "best.pth"))
            psnr_max = psnr
        lr_scheduler.step()


# ######################################################################################################


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train a reverseISP on pairs of RAW - RGB image patches.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU used to train.")
    parser.add_argument("-c", "--cfg", type=str, default="", help="config file")
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    device = torch.device(f"cuda:{cuda_no}")
    torch.cuda.set_device(device)

    dataset, cfg_sample, cfg_train = load_cfg(args.cfg)

    folder_data_train = DatasetSamples(dataset, cfg_sample, cfg_train, mode="train")
    folder_data_test = DatasetSamples(dataset, cfg_sample, cfg_train, mode="test")

    trainloader = DataLoader(
        folder_data_train, batch_size=cfg_train["batch_size"], shuffle=True, num_workers=8
    )
    testloader = DataLoader(
        folder_data_test, batch_size=cfg_train["batch_size"], shuffle=False, num_workers=8
    )

    model = ReRAW(
        in_size=3,
        out_size=4,
        target_size=cfg_train["target_size"],
        hidden_size=cfg_train["hidden_size"],
        n_layers=cfg_train["depth"],
        gammas=cfg_train["gammas"]
    )

    model.cuda()

    TIME_STAMP = int(time.time())
    project_name = f"{TIME_STAMP}"

    # make workspace folders
    save_path = cfg_train["save_path"] + project_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Saving model at:", save_path)
    writer = SummaryWriter(cfg_train["tensorboard_path"] + project_name)

    # save the used config in the workspace folder
    save_cfg(cfg_train, cfg_sample, dataset, os.path.join(save_path, "cfg.py"))

    train(model, trainloader, testloader, save_path, cfg_train, writer)