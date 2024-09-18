import math
import os.path
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from shared.common_utils import export_point_cloud, check_dir

class nerf2nvd_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(nerf2nvd_dataset, self).__init__()
        self.data_root = Path(v_conf["root"])
        self.mode = v_training_mode
        self.conf = v_conf

        # Img
        self.img = np.asarray(Image.open(self.data_root/ "0_colors.png"))[:,:,:3]

        # Voronoi
        # voronoi = np.frombuffer(open(self.data_root/"00001817_0.bin", "rb").read(), dtype=np.uint8)
        # shifts = np.arange(8)
        # flags = (voronoi[:,None] & (1 << shifts)[None,:]).reshape(-1) > 0
        # self.voronoi = flags.reshape(256,256,256)

        # UDF and Voronoi
        with h5py.File(self.data_root/"training.h5", "r") as f:
            udf = np.asarray(f["features"], dtype=np.float32)[0,...,0:1]/65535*2.
            flags = np.asarray(f["flags"][0])[...,None]>0

        self.udf = torch.from_numpy(udf).permute(3,0,1,2).contiguous()
        self.voronoi_flags = torch.from_numpy(flags).permute(3,0,1,2).contiguous()
        self.length = 100

        self.query_points = np.reshape(np.stack(np.meshgrid(
            np.arange(256), np.arange(256), np.arange(256), indexing='ij'),axis=-1), (-1,3)
        ).astype(np.float32)
        self.query_points = torch.from_numpy(self.query_points / 255 * 2 - 1)

        # Debug
        if 0:
            vertices = np.stack(np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij'),axis=-1)
            vertices = vertices / 255 * 2 - 1
            vertices = vertices[self.voronoi]
            export_point_cloud("1.ply", vertices)


    def __len__(self):
        return self.length if self.mode=="training" else 1

    def __getitem__(self, idx):
        return self.udf, self.voronoi_flags, self.query_points


