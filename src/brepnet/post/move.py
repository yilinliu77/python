import os, shutil
from tqdm import tqdm
import argparse, sys

data_root = r"/mnt/d/img2brep/0924_0914_dl8_ds256_context_kl_v5_test"

def move(data_root):
    save_root = data_root + "_out"
    # save_root = r"/mnt/d/img2brep/0909_test_out"
    os.makedirs(save_root, exist_ok=True)
    all_files = os.listdir(data_root)
    all_files.sort()

    folder_names = []
    for filename in tqdm(all_files):
        if filename.endswith("_feature.npz"):
            folder_names = filename.split("_")[0]
            tgt_path = os.path.join(save_root, folder_names)
            os.makedirs(tgt_path, exist_ok=True)
            shutil.copy(os.path.join(data_root, folder_names + ".npz"), os.path.join(tgt_path, "data.npz"))


def count_success(data_root):
    data_root = data_root+"_out"
    os.makedirs(failed_root, exist_ok=True)
    all_files = os.listdir(data_root)
    all_files.sort()

    for filename in tqdm(all_files):
        if os.path.exists(os.path.join(data_root, filename, "recon_brep.step")):
            continue
        else:
            pass
            shutil.copytree(os.path.join(data_root, filename), os.path.join(failed_root, filename))


if __name__ == '__main__':
    data_root = sys.argv[1]
    move(data_root)
    # count_success(data_root)
