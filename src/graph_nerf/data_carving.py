import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

scannet_root = r"D:\DATASET\SCANNET\scannet_pre\scans\scene0000_00"
color_dir = os.path.join(scannet_root, "color")
instance_dir = os.path.join(scannet_root, "instance")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def sam_detect():
    sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_img)
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_img)
    show_anns(masks)
    plt.axis('off')
    plt.show(block=True)

if __name__ == '__main__':
    num_img = len(os.listdir(color_dir))
    for i_img in range(num_img):
        rgb_img = cv2.imread(os.path.join(color_dir, "{:06d}.jpg".format(i_img*100)), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        seg_img = cv2.imread(os.path.join(instance_dir, "{:06d}.png".format(i_img*100)), cv2.IMREAD_UNCHANGED)



        pass

    pass