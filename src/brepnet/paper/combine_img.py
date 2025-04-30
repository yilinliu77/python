from pathlib import Path

from PIL import Image
import os

import numpy as np
import random, sys
from rectpack import newPacker
from tqdm import tqdm


def crop_object(image, img_name, save_result=False, save_directory=None):
    image = image.crop((50, 50, image.size[0] - 50, image.size[1] - 50))
    target_size = image.size
    background_threshold = 250

    img_array = np.array(image)[:, :, :3]
    non_white_mask = np.all(img_array < background_threshold, axis=-1)
    coords = np.argwhere(non_white_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    assert target_size[0] == target_size[1]
    center = target_size[0] // 2
    radius_x = x_max - center + center // 10
    radius_y = max(center - y_min, y_max - center) + center // 20

    image = image.crop((center - radius_x, center - radius_y, center + radius_x, center + radius_y))
    if save_result:
        if save_directory is not None:
            image.save(os.path.join(save_directory, os.path.basename(img_name)))
        else:
            image.save(os.path.join("cropped_raw", os.path.basename(img_name)))
    return image


def fix_crop_object(image, v_size=(0.7, 0.7)):
    target_size = image.size

    x_start = int(target_size[0] * (0.5 - v_size[0] / 2))
    x_end = int(target_size[0] * (0.5 + v_size[0] / 2))
    y_start = int(target_size[1] * (0.5 - v_size[1] / 2))
    y_end = int(target_size[1] * (0.5 + v_size[1] / 2))
    image = image.crop((x_start, y_start, x_end, y_end))
    return image


def combine_with_same_size(images1, row: int, col: int, cell_size: tuple, intervals: tuple):
    scaled_imgs = []
    for img in images1:
        current_width, current_height = img.size
        if current_width < cell_size[0] and current_height < cell_size[1]:
            scale = min(cell_size[0] / current_width, cell_size[1] / current_height)
        else:
            scale = 1.0 / max(cell_size[0] / current_width, cell_size[1] / current_height)

        new_width = int(current_width * scale)
        new_height = int(current_height * scale)

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        scaled_imgs.append(img)

    combined_width = col * cell_size[1] + (col - 1) * intervals[1]
    combined_height = row * cell_size[0] + (row - 1) * intervals[0]

    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    for i, img in enumerate(tqdm(scaled_imgs)):
        ri = i // col
        ci = i % col

        upper_left = (
            ci * (intervals[1] + cell_size[1]),
            ri * (intervals[0] + cell_size[0])
        )
        combined_image.paste(img, upper_left)

    return combined_image


def pack_img(images, bin):
    packer = newPacker(rotation=False)

    # Add rectangles with IDs corresponding to their index
    for i, img in enumerate(images):
        packer.add_rect(img.size[0], img.size[1], rid=i)

    # Add the bin
    packer.add_bin(*bin)
    packer.pack()

    # Create a new image to paste packed images
    packed_img = Image.new('RGB', bin, (255, 255, 255))

    # Map rectangle positions back to their images using 'rid'
    for rect in packer.rect_list():
        bin_id, x, y, w, h, rid = rect
        if rid == 32:
            pass
        img = images[rid]  # Get the correct image using its ID
        upper_left = (x, y)
        packed_img.paste(img, upper_left)

    return packed_img


if __name__ == "__main__":
    row_number = 5
    col_number = 15
    imgs_root1 = Path("D:/brepnet/paper/uncond/v2/1127_730_li_270k_1gpu_75/1127_730_li_270k_1gpu_75_suce_imgs")
    imgs_root2 = Path("D:/brepnet/paper/uncond/v2/1127_730_li_270k_1gpu_75/1127_730_li_270k_1gpu_75_fail_imgs")

    print("Read imgs")
    crop_ratio = 1.
    imgs1 = [fix_crop_object(Image.open(str(img)), v_size=(crop_ratio, crop_ratio)) for img in tqdm(imgs_root1.iterdir())]
    imgs2 = [fix_crop_object(Image.open(str(img)), v_size=(crop_ratio, crop_ratio)) for img in tqdm(imgs_root2.iterdir())]

    imgs = imgs1 + imgs2

    print("Failure start from {}".format(len(imgs1)))
    print("Combine imgs")
    combined_img = combine_with_same_size(
        imgs, row=row_number, col=col_number, cell_size=(4000, 4000), intervals=(0, 0))

    combined_img.save("combined.png")
    downsampled_size = (combined_img.width // 10, combined_img.height // 10)
    downsampled_combined_img = combined_img.resize(downsampled_size, Image.Resampling.BICUBIC)
    downsampled_combined_img.save("dowmsampled_combined.png")