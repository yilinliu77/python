import os
import argparse
import glob

from src.brepnet.post.construct_brep import *
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument('--prefix', type=str, default="")
    args = parser.parse_args()
    data_root = args.data_root
    out_root = args.out_root
    prefix = args.prefix

    if prefix != "":
        test_construct_brep(data_root, out_root, prefix, True, True)

    folders = os.listdir(data_root)
    folders.sort()
    # Load cad data
    for folder_name in tqdm(folders):
        construct_brep_from_datanpz(data_root, out_root, folder_name)
