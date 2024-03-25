import json
import os, shutil

import cv2
import numpy as np

from tqdm import tqdm

from shared.common_utils import check_dir
from src.neural_recon.colmap_io import read_dataset

input_root = r"G:/Projects/img2brep/data/00016422"
output_root = r"G:/Projects/img2brep/data/t_neus2_sparse/00016422"

if __name__ == '__main__':
    check_dir(output_root)
    dataset, _ = read_dataset(input_root, None)
    validation_split = list(range(0, len(dataset), len(dataset)//10))
    train_split = list(set(range(0, len(dataset))) - set(validation_split))
    output_json={
        "w": 800,
        "h": 800,
        "aabb_scale": 1,
        "scale": 0.5,
        "offset": [0.25, 0.25, 0.25],
        "from_na": True,
    }

    cameras = []
    for item in tqdm(np.asarray(dataset)[[33,37,39]]):
        R = item.extrinsic[:3, :3]
        t = item.extrinsic[:3, 3:4]
        # Following code are from nerfstudio
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        # c2w[0:3, 1:3] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1
        intrinsic = item.intrinsic.copy()
        intrinsic[0, :] *= item.img_size[0]
        intrinsic[1, :] *= item.img_size[1]
        cameras.append({
            "file_path": item.img_name+".png",
            "intrinsic_matrix": intrinsic.tolist(),
            "transform_matrix": c2w.tolist(),
        })
        img = cv2.imread(item.img_path, cv2.IMREAD_COLOR)[:,:,:3]

        mask_img = np.logical_not(np.all(img == 0, axis=2))
        mask_img = mask_img.astype(np.uint8)*255
        cv2.imwrite(os.path.join(output_root, item.img_name+".png"),
                    np.concatenate([img, mask_img[...,None]], axis=2))

    output_json["frames"] = cameras
    json.dump(output_json, open(os.path.join(output_root, "transform.json"), "w"))
    pass