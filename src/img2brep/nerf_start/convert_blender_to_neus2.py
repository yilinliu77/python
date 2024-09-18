import json
import os, shutil

import cv2
import numpy as np

from tqdm import tqdm

from shared.common_utils import check_dir
from src.neural_recon.colmap_io import read_dataset

input_root = r"G:/Projects/img2brep/data/0planar_shapes/00008300"
output_root = r"G:/Projects/img2brep/data/t_neus2_sparse/00008300"

if __name__ == '__main__':
    check_dir(output_root)
    trained_poses = json.loads(open(os.path.join(input_root, "transforms_train.json"), "r").read())
    validation_poses = json.loads(open(os.path.join(input_root, "transforms_val.json"), "r").read())
    output_json={
        "w": 800,
        "h": 800,
        "aabb_scale": 1,
        "scale": 0.5,
        "offset": [0.25, 0.25, 0.25],
        "from_na": True,
    }

    cameras = []
    for item in tqdm(np.asarray(trained_poses["frames"])[[33,37,39]]):
        intrinsic = item["camera_intrinsics"]

        scale_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        extrinsic = np.asarray(item["transform_matrix"])
        extrinsic = scale_matrix @ np.linalg.inv(extrinsic)
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3:4]
        # Following code are from nerfstudio
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c).tolist()

        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        cameras.append({
            "file_path": item["file_path"].split("/")[-1]+".png",
            "intrinsic_matrix": intrinsic,
            "transform_matrix": c2w,
        })
        img = cv2.imread(os.path.join(input_root,"train_img",cameras[-1]["file_path"]), cv2.IMREAD_ANYCOLOR)[:,:,:3]

        mask_img = np.logical_not(np.all(img < 2, axis=2))
        mask_img = mask_img.astype(np.uint8)*255
        cv2.imwrite(os.path.join(output_root, cameras[-1]["file_path"]),
                    np.concatenate([img, mask_img[...,None]], axis=2))

    output_json["frames"] = cameras
    json.dump(output_json, open(os.path.join(output_root, "transform.json"), "w"))
    pass