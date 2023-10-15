import os, h5py
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    with h5py.File(r"/mnt/d/GSP/GSP_v7_100k/training.h5") as f:
        num_items = f["names"].shape[0]
        num_valid = 0
        for i in range(num_items):
            name = "{:08d}".format(f["names"][i])
            id = f["ids"][i]
            if not os.path.exists("/mnt/d/GSP/GSP_v7_100k_p/pointcloud/{}_{}.ply".format(name, id)):
                print("{}_{}.ply".format(name, id))
            else:
                num_valid+=1
    pass