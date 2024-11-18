import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import rawpy


def load_raw(file_path):
    with rawpy.imread(file_path) as raw:
        raw_image = raw.raw_image_visible

    height, width = raw_image.shape
    r = raw_image[0:height:2, 0:width:2]
    g1 = raw_image[0:height:2, 1:width:2]
    g2 = raw_image[1:height:2, 0:width:2]
    b = raw_image[1:height:2, 1:width:2]

    # Stack the channels into a single array
    rggb_image = np.stack((r, g1, g2, b), axis=-1)
    return rggb_image


def convert_image(
    model, img_rgb, sample_size=(64, 64), context_size=(128, 128), gammas=[1], batch_size=64
):
    model.eval()
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    img_rggb_h, img_rggb_w = img_rgb_h // 2, img_rgb_w // 2
    sample_size_half = [sample_s // 2 for sample_s in sample_size]
    out_rggb = np.zeros((img_rggb_h, img_rggb_w, 4), dtype=np.float32)
    context = cv2.resize(img_rgb, context_size, interpolation=cv2.INTER_AREA)

    r, c = 0, 0
    indexes = []
    samples = []
    global_imgs = []

    # image needs to be extended by 1 pixel since the reverseISP model takes in a 66x66 patch and outpus a 32x32 patch
    img_rgb_extended = cv2.copyMakeBorder(img_rgb, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    # the model is convolved over the image
    while r < img_rgb_h:
        c = 0
        while c < img_rgb_w:
            if c + sample_size[1] > img_rgb_w:
                c = img_rgb_w - sample_size[1]
            if r + sample_size[0] > img_rgb_h:
                r = img_rgb_h - sample_size[0]

            # take a 66x66 patch
            sample = img_rgb_extended[r : r + sample_size[0] + 2, c : c + sample_size[1] + 2]

            samples.append(sample)
            global_imgs.append(context)
            indexes.append([r // 2, c // 2])

            # once the full batch has been filled, convert all patches
            if len(indexes) == batch_size or (
                c + sample_size[1] >= img_rgb_w and r + sample_size[0] >= img_rgb_h
            ):
                # compute
                samples_stacked = np.transpose(np.stack(samples), (0, 3, 1, 2))
                global_imgs_stacked = np.transpose(np.stack(global_imgs), (0, 3, 1, 2))

                samples = torch.from_numpy(samples_stacked).cuda()
                global_imgs = torch.from_numpy(global_imgs_stacked).cuda()

                y, outputs, _ = model(samples, global_imgs)

                y = y.detach().cpu().numpy()
                y = y.transpose(0, 2, 3, 1)

                # stich image
                for i, idx in enumerate(indexes):
                    h, w = idx
                    out_rggb[h : h + sample_size_half[0], w : w + sample_size_half[1]] = y[i]

                indexes = []
                samples = []
                global_imgs = []

            c += sample_size[1]
        r += sample_size[0]

    out_rggb = np.clip(out_rggb, 0, 1)
    return out_rggb


def center_crop(image, image_size_h, image_size_w):
    image_height, image_width = image.shape[:2]
    top = (image_height - image_size_h) // 2
    left = (image_width - image_size_w) // 2
    patch_image = image[top : top + image_size_h, left : left + image_size_w, ...]
    return patch_image


def test_patches(model, dataloader, gammas=None):
    psnrs = []
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"------->>>> Testing:")
                sample, targets, context, target = (
                    data[0].cuda(),
                    data[1].cuda(),
                    data[2].cuda(),
                    data[3].cuda(),
                )
                y, outputs, _ = model(sample, context)

                psnrs += compute_psnr_patches(y, target)

    psnr_mean = np.mean(psnrs)
    print("-> Mean PSNR/patch: ", np.round(psnr_mean, 2))

    return psnr_mean


def compute_psnr_patches(outputs, targets):
    psnrs = []
    outputs = outputs.detach().cpu().numpy()
    outputs = outputs.transpose(0, 2, 3, 1)

    targets = targets.detach().cpu().numpy()
    targets = targets.transpose(0, 2, 3, 1)

    for i in range(targets.shape[0]):
        psnrs.append(-10 * np.log10(np.mean((targets[i] - outputs[i]) ** 2)))

    return psnrs


# This function takes a large crop of the image of size*scale
# that definitely contains the 'location' pixel.
def large_crop(image, location, scale, resize):
    h, w = image.shape[:2]
    h_scaled_half = int(h * scale / 2)
    w_scaled_half = int(w * scale / 2)
    h_scaled = int(h * scale)
    w_scaled = int(w * scale)
    x, y = location

    if x - w_scaled_half < 0:
        x1 = 0
        x2 = w_scaled
    elif x + w_scaled_half > w:
        x2 = w
        x1 = x2 - w_scaled
    else:
        x1 = x - w_scaled_half
        x2 = x + w_scaled_half

    if y - h_scaled_half < 0:
        y1 = 0
        y2 = h_scaled
    elif y + h_scaled_half > h:
        y2 = h
        y1 = y2 - h_scaled
    else:
        y1 = y - h_scaled_half
        y2 = y + h_scaled_half

    crop = image[y1:y2, x1:x2]
    resized = cv2.resize(crop, resize, interpolation=cv2.INTER_AREA)

    return resized


def convert_to_rgb(img_rggb, gamma=1):
    if img_rggb.shape[0] == 4:
        img_rggb = np.transpose(img_rggb, (1, 2, 0))

    r, g1, g2, b = np.split(img_rggb, 4, axis=-1)
    g = (g1 + g2) / 2.0  # Average the two green channels
    img = np.concatenate([r, g, b], axis=-1).astype(np.float32)

    img = np.clip(img, 0, 1 - 1 / 2**8)
    if gamma:
        img = img**gamma
    img_rggb_to_rgb = np.asarray(img * 2**8, dtype=np.uint8)
    return img_rggb_to_rgb


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print("Saved model:", save_path)


def save_cfg(cfg_train, cfg_sample, dataset, save_cfg_path):
    with open(save_cfg_path, "w") as file:
        # Write each dictionary to the file
        file.write("dataset = " + str(dataset) + "\n")
        file.write("cfg_sample = " + str(cfg_sample) + "\n")
        file.write("cfg_train = " + str(cfg_train) + "\n")


def load_cfg(cfg_path):
    variables = {}
    with open(cfg_path) as file:
        exec(file.read(), variables)

    dataset = variables["dataset"]
    cfg_train = variables["cfg_train"]
    cfg_sample = variables["cfg_sample"]

    return dataset, cfg_sample, cfg_train

