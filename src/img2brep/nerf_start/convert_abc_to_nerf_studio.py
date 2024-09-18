import json
import os, shutil

import cv2
import numpy as np

from tqdm import tqdm

from shared.common_utils import check_dir
from src.neural_recon.colmap_io import read_dataset

input_root = r"G:/Projects/img2brep/data/00016422"
output_root = r"G:/Projects/img2brep/data/t_nerfstudio"

if __name__ == '__main__':
    check_dir(output_root)
    dataset, _ = read_dataset(input_root, None)
    validation_split = list(range(0, len(dataset), len(dataset)//10))
    train_split = list(set(range(0, len(dataset))) - set(validation_split))
    output_json={
        "camera_model": "OPENCV",
        "train_filenames": [dataset[id].img_name+".jpg" for id in train_split],
        "val_filenames": [dataset[id].img_name+".jpg" for id in validation_split],
        "test_filenames": [dataset[id].img_name+".jpg" for id in validation_split],
    }

    cameras = []
    for item in tqdm(dataset):
        R = item.extrinsic[:3, :3]
        t = item.extrinsic[:3, 3:4]
        # Following code are from nerfstudio
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        cameras.append({
            "file_path": item.img_name+".jpg",
            "w": item.img_size[0],
            "h": item.img_size[1],
            "fl_x": item.intrinsic[0,0]*item.img_size[0],
            "fl_y": item.intrinsic[1,1]*item.img_size[1],
            "cx": item.intrinsic[0,2]*item.img_size[0],
            "cy": item.intrinsic[1,2]*item.img_size[1],
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "k4": 0,
            "p1": 0,
            "p2": 0,
            "transform_matrix": c2w.tolist(),
            "mask_path": item.img_name+"_mask.jpg"
        })
        img = cv2.imread(item.img_path, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(output_root, item.img_name+".jpg"), img)

        mask_img = np.logical_not(np.all(img == 0, axis=2))
        mask_img = mask_img.astype(np.uint8)*255
        cv2.imwrite(os.path.join(output_root, item.img_name+"_mask.jpg"), mask_img)

    output_json["frames"] = cameras
    json.dump(output_json, open(os.path.join(output_root, "transforms.json"), "w"))
    pass