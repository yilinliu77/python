import cv2
import open3d as o3d
import scipy
import trimesh
from tqdm import tqdm
import os
import numpy as np

from shared.common_utils import safe_check_dir, check_dir

if __name__ == '__main__':
    # root_dir = r"G:/Dataset/GSP/Results/viz_output/test2/Bright_color_imgs"
    # output_dir = r"G:/Dataset/GSP/Results/viz_output/test2/vis6"
    # ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis6.txt").readlines()]

    # root_dir = r"G:/Dataset/GSP/Results/viz_output/test2/Bright_color_random_imgs"
    # output_dir = r"G:/Dataset/GSP/Results/viz_output/test2/vis_random6"
    # ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis_random6.txt").readlines()]

    root_dir = r"G:/Dataset/GSP/Results/viz_output/0120_random_mesh_imgs"
    output_dir = r"G:/Dataset/GSP/Results/viz_output/0120_random_merged"
    ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis_random.txt").readlines()]

    # root_dir = r"G:/Dataset/GSP/Results/viz_output/0120_mesh_imgs"
    # output_dir = r"G:/Dataset/GSP/Results/viz_output/0120_merged"
    # ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis31.txt").readlines()]

    # ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis_random.txt").readlines()]
    # ids = [file.strip() for file in open(r"G:/Dataset/GSP/List/viz_ids_small2.txt").readlines()]
    safe_check_dir(output_dir)

    files = sorted([file[:8] for file in os.listdir(root_dir) if file.endswith("_gt.png")])

    total_imgs = []
    num_total_img = 0
    for prefix in tqdm(files):
        if prefix not in ids:
            continue
        gt_png = cv2.imread(os.path.join(root_dir, prefix + "_gt.png"))
        gt_wire_png = cv2.imread(os.path.join(root_dir, prefix + "_gt_curve.png"))
        complex_png = cv2.imread(os.path.join(root_dir, prefix + "_complex.png"))
        complex_wire_png = cv2.imread(os.path.join(root_dir, prefix + "_complex_curve.png"))
        hp_png = cv2.imread(os.path.join(root_dir, prefix + "_hp.png"))
        hp_wire_png = cv2.imread(os.path.join(root_dir, prefix + "_hp_curve.png"))
        sed_png = cv2.imread(os.path.join(root_dir, prefix + "_sed.png"))
        sed_wire_png = cv2.imread(os.path.join(root_dir, prefix + "_sed_curve.png"))
        ours_png = cv2.imread(os.path.join(root_dir, prefix + "_ours.png"))
        ours_wire_png = cv2.imread(os.path.join(root_dir, prefix + "_ours_curve.png"))

        paddingy = 1
        paddingx = 1
        gt_png = gt_png[paddingy:-paddingy, paddingx:-paddingx, :]
        gt_wire_png = gt_wire_png[paddingy:-paddingy, paddingx:-paddingx, :]
        complex_png = complex_png[paddingy:-paddingy, paddingx:-paddingx, :]
        complex_wire_png = complex_wire_png[paddingy:-paddingy, paddingx:-paddingx, :]
        hp_png = hp_png[paddingy:-paddingy, paddingx:-paddingx, :]
        hp_wire_png = hp_wire_png[paddingy:-paddingy, paddingx:-paddingx, :]
        sed_png = sed_png[paddingy:-paddingy, paddingx:-paddingx, :]
        sed_wire_png = sed_wire_png[paddingy:-paddingy, paddingx:-paddingx, :]
        ours_png = ours_png[paddingy:-paddingy, paddingx:-paddingx, :]
        ours_wire_png = ours_wire_png[paddingy:-paddingy, paddingx:-paddingx, :]

        # gt_png = np.pad(
        #     gt_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
        #     constant_values=255)
        # complex_png = np.pad(
        #     complex_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
        #     constant_values=255)
        # hp_png = np.pad(
        #     hp_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
        #     constant_values=255)
        # sed_png = np.pad(
        #     sed_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
        #     constant_values=255)
        # ours_png = np.pad(
        #     ours_png, pad_width=((0, 0), (0, 0), (0, 0)), mode="constant",
        #     constant_values=255)

        single_img = np.concatenate([
            np.concatenate([complex_png, complex_wire_png], axis=0),
            np.concatenate([hp_png, hp_wire_png], axis=0),
            np.concatenate([sed_png, sed_wire_png], axis=0),
            np.concatenate([ours_png, ours_wire_png], axis=0),
            np.concatenate([gt_png, gt_wire_png], axis=0)
        ], axis=1)

        cv2.imwrite(os.path.join(output_dir, prefix + "_total.png"), single_img)
        total_imgs.append(single_img)

        if len(total_imgs) >= 5:
            total_img = np.concatenate(total_imgs, axis=0)
            cv2.imwrite(os.path.join(output_dir, "{:03d}.jpg".format(num_total_img)), total_img)
            num_total_img+=1
            total_imgs = []
        # break

    if len(total_imgs) !=0 :
        total_img = np.concatenate(total_imgs, axis=0)
        cv2.imwrite(os.path.join(output_dir, "{:03d}.jpg".format(num_total_img)), total_img)
