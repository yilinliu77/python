import os
import shutil

import ray
import glob
from tqdm import tqdm
from OCC.Extend.DataExchange import read_step_file, write_stl_file
import argparse


# data_root = r"E:\data\img2brep\test"
# out_root = r"E:\data\img2brep\test"


def step2stl(step_folder, output_folder):
    try:
        step_files = glob.glob(os.path.join(step_folder, "*.step"))
        if len(step_files) == 0:
            return
        step_file = step_files[0]
        stl_file = os.path.join(output_folder, "mesh.stl")
        shape = read_step_file(step_file)
        os.makedirs(output_folder, exist_ok=True)
        write_stl_file(shape, stl_file, linear_deflection=0.01, angular_deflection=0.5)
    except Exception as e:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        pass


step2stl_remote = ray.remote(step2stl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    args = parser.parse_args()
    data_root = args.data_root
    out_root = args.out_root

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    all_folders = os.listdir(data_root)

    ray.init()
    futures = []
    for folder in tqdm(all_folders):
        step_folder = os.path.join(data_root, folder)
        output_folder = os.path.join(out_root, folder)
        futures.append(step2stl_remote.remote(step_folder, output_folder))
    for f in tqdm(futures):
        ray.get(f)
    ray.shutdown()
