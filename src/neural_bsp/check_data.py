import os, h5py
import open3d as o3d
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    with h5py.File(r"G:\Dataset\GSP\v13_100k/training.h5") as f:
        num_items = f["point_flags"].shape[0]
        num_patch = f["point_flags"].shape[1]
        for i in tqdm(range(num_items)):
            if (f["point_flags"][i] == 0).all(axis=1).any():
                print("Error: {}".format(i))
    pass