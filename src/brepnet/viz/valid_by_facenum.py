import os
from pathlib import Path
import random

from tqdm import tqdm
import argparse
import numpy as np
import shutil
import ray
import glob

from src.brepnet.eval.check_valid import *
from src.brepnet.post.utils import *
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_SOLID
import trimesh

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def check_face_num_and_validity(data_root, post_root, folder, data_npz_name="data.npz"):
    data_npz = np.load(os.path.join(data_root, folder, data_npz_name))
    if 'sample_points_faces' in data_npz and 'edge_face_connectivity' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
    elif 'pred_face' in data_npz:
        face_points = data_npz['pred_face']
    else:
        raise Exception("check keys in data.npz")
    face_num = face_points.shape[0]
    try:
        step_file_list = load_data_with_prefix(os.path.join(post_root, folder), ".step")
        if len(step_file_list) == 0:
            return {"is_valid_solid": False, "num_faces": face_num}
        return {"is_valid_solid": check_step_valid_soild(step_file_list[0]), "num_faces": face_num}
    except:
        return {"is_valid_solid": False, "num_faces": face_num}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--post_root", type=str, required=True)
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--data_npz_name", type=str, default="data.npz")
    args = parser.parse_args()

    data_root = args.data_root
    post_root = args.post_root
    data_npz_name = args.data_npz_name

    ray.init(local_mode=False)
    check_face_num_and_validity_remote = ray.remote(check_face_num_and_validity)
    # load all generated folder
    all_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    futures = []
    for folder in tqdm(all_folders):
        futures.append(check_face_num_and_validity_remote.remote(data_root, post_root, folder, data_npz_name))

    results = []
    for future in tqdm(futures):
        results.append(ray.get(future))

    # viz the valid_ratio-face_num distribution (for all samples, not only valid)
    result_dict = {}
    for face_num in range(3, 31):
        result_dict[face_num] = []

    for item in results:
        face_num = item["num_faces"]
        is_valid = item["is_valid_solid"]
        if face_num not in result_dict:
            continue
        result_dict[face_num].append(is_valid)

    plt.figure(figsize=(12, 6))
    face_num = [str(i).zfill(2) for i in range(3, 31)]
    sample_num = [len(result_dict[i]) for i in range(3, 31)]
    sns.barplot(x=face_num, y=sample_num)
    plt.title('Sample num by Faces num')
    plt.xlabel('Faces num')
    plt.xticks(rotation=45)
    plt.ylabel('Sample num')
    plt.tight_layout()
    plt.savefig(data_root + "_sample_num_by_faces_num.png")

    plt.figure(figsize=(12, 6))
    face_num = [str(i).zfill(2) for i in range(3, 31)]
    valid_ratio = []
    for i in range(3, 31):
        if len(result_dict[i]) == 0:
            face_num.remove(str(i).zfill(2))
        else:
            valid_ratio.append(np.sum(result_dict[i]) / len(result_dict[i]))
    sns.barplot(x=face_num, y=valid_ratio)
    plt.title('Valid ratio by Faces num')
    plt.ylim(0, 1)
    plt.xlabel('Faces num')
    plt.xticks(rotation=45)
    plt.ylabel('Valid ratio')
    plt.tight_layout()
    plt.savefig(data_root + "_valid_ratio_by_faces_num.png")
