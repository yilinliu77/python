import os
import shutil

import ray
import glob
from tqdm import tqdm
from OCC.Extend.DataExchange import read_step_file, write_stl_file
import argparse


# data_root = r"E:\data\img2brep\test"
# out_root = r"E:\data\img2brep\test"


def step2stl(data_root, step_folders, output_root):
    for step_folder in step_folders:
        output_folder = str(os.path.join(output_root, os.path.basename(step_folder)))
        try:
            step_files = glob.glob(os.path.join(data_root, step_folder, "*.step"))
            if len(step_files) == 0:
                continue
            step_file = step_files[0]
            shape = read_step_file(step_file, as_compound=False, verbosity=False)
            stl_file = os.path.join(output_folder, "mesh.stl")
            os.makedirs(output_folder, exist_ok=True)
            write_stl_file(shape, stl_file, linear_deflection=0.9, angular_deflection=0.5)
        except Exception as e:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            pass


step2stl_remote = ray.remote(step2stl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--list", type=str, default=None, help="list file in the training set")
    args = parser.parse_args()
    data_root = args.data_root
    out_root = args.out_root
    if args.list is not None:
        with open(args.list, "r") as f:
            all_folders = f.read().splitlines()
    else:
        all_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    os.makedirs(out_root, exist_ok=True)

    if True:
        all_folders = list(set(all_folders) - set(os.listdir(out_root)))

    print(f"Total {len(all_folders)} folders to process")
    
    ray.init(local_mode=False)
    futures = []
    batch_size = 1
    for i in tqdm(range(0, len(all_folders), batch_size)):
        step_folders = all_folders[i:i + batch_size]
        futures.append(step2stl_remote.remote(data_root, step_folders, out_root))
    for f in tqdm(futures):
        ray.get(f)
    ray.shutdown()
