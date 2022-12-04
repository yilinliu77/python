import math
import os
import pickle

import cv2
import hydra
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn.functional import grid_sample

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

import original_warp
from matplotlib import pyplot as plt

def generate_sample_after_homography(v_img: torch.Tensor,
                                     v_num_sample: int,
                                     v_degree_of_freedom: int,
                                     v_noise_t, v_noise_h,
                                     v_crop_size):
    img_height = v_img.shape[1]
    img_width = v_img.shape[2]

    warp_pert_all = torch.zeros(v_num_sample, v_degree_of_freedom)
    trans_pert = [(0, 0)] + [(x, y) for x in (-v_noise_t, v_noise_t)
                             for y in (-v_noise_t, v_noise_t)]

    def create_random_perturbation():
        warp_pert = torch.randn(v_degree_of_freedom) * v_noise_h
        warp_pert[0] += trans_pert[i][0]
        warp_pert[1] += trans_pert[i][1]
        return warp_pert

    for i in range(v_num_sample):
        warp_pert = create_random_perturbation()
        while not original_warp.check_corners_in_range(img_height, img_width, v_crop_size, v_num_sample, warp_pert[None]):
            warp_pert = create_random_perturbation()
        warp_pert_all[i] = warp_pert
    if True:
        warp_pert_all[0] = 0
    # create warped image patches
    xy_grid = original_warp.get_normalized_pixel_grid_crop(img_height, img_width, v_crop_size, v_num_sample, "cpu")  # [B,HW,2]
    xy_grid_warped = original_warp.warp_grid(xy_grid, warp_pert_all)
    xy_grid_warped = xy_grid_warped.view([v_num_sample, v_crop_size, v_crop_size, 2])
    xy_grid_warped = torch.stack([xy_grid_warped[..., 0] * max(img_height, img_width) / img_width,
                                  xy_grid_warped[..., 1] * max(img_height, img_width) / img_height], dim=-1)
    image_raw_batch = v_img.repeat(v_num_sample, 1, 1, 1)
    image_pert_all = grid_sample(image_raw_batch, xy_grid_warped, align_corners=False)
    return image_pert_all

class Single_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_img_path, v_img_width, v_img_height, v_crop_size, v_mode):
        super(Single_img_dataset, self).__init__()
        self.trainer_mode = v_mode
        img_raw = Image.open(v_img_path)
        img_raw_resized = img_raw
        if img_raw.width!=v_img_width or img_raw.height!=v_img_height:
            img_raw_resized = img_raw.resize((v_img_width, v_img_height))
        self.img_raw = np.asarray(img_raw_resized).copy()
        img_raw_resized.close()
        img_raw.close()
        self.img_raw = to_tensor(self.img_raw)
        self.trained_imgs = generate_sample_after_homography(self.img_raw, 5, 8, 0.5, 0.5, v_crop_size)

        # Debug
        # for i in range(5):
        #     img = self.trained_imgs[i].permute([1,2,0]).numpy()
        #     # self.hydra_conf["trainer"]["output"]
        #     plt.imshow(img)
        #     plt.show()

        pass

    def __getitem__(self, index):
        return self.trained_imgs, self.img_raw

    def __len__(self):
        if self.trainer_mode == "training":
            return 5000
        else:
            return 1
