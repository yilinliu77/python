import cv2
import open3d as o3d
import scipy
import trimesh
from tqdm import tqdm
import os
import numpy as np

from shared.common_utils import safe_check_dir, check_dir

if __name__ == '__main__':
    root_dir = r"G:/Dataset/GSP/viz_output/out"
    output_dir = r"G:/Dataset/GSP/viz_output/out2"
    safe_check_dir(output_dir)

    files = sorted([file[:8] for file in os.listdir(root_dir) if file.endswith("_gt.png")])

    total_imgs = []
    num_total_img = 0
    for prefix in tqdm(files):
        gt_png = cv2.imread(os.path.join(root_dir, prefix + "_gt.png"))
        complex_png = cv2.imread(os.path.join(root_dir, prefix + "_complex.png"))
        hp_png = cv2.imread(os.path.join(root_dir, prefix + "_hp.png"))
        sed_png = cv2.imread(os.path.join(root_dir, prefix + "_sed.png"))
        ours_png = cv2.imread(os.path.join(root_dir, prefix + "_ours.png"))

        gt_png = np.pad(
            gt_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
            constant_values=180)
        complex_png = np.pad(
            complex_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
            constant_values=180)
        hp_png = np.pad(
            hp_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
            constant_values=180)
        sed_png = np.pad(
            sed_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
            constant_values=180)
        ours_png = np.pad(
            ours_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
            constant_values=180)

        single_img = np.concatenate([complex_png, hp_png, sed_png, ours_png, gt_png], axis=1)

        cv2.imwrite(os.path.join(output_dir, prefix + "_total.png"), single_img)
        total_imgs.append(single_img)

        if len(total_imgs) >= 6:
            total_img = np.concatenate(total_imgs, axis=0)
            cv2.imwrite(os.path.join(output_dir, "{:03d}.jpg".format(num_total_img)), total_img)
            num_total_img+=1
            total_imgs = []

    total_img = np.concatenate(total_imgs, axis=0)
    cv2.imwrite(os.path.join(output_dir, "{:03d}.jpg".format(num_total_img)), total_img)
