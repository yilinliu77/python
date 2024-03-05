import math
import os.path
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from shared.common_utils import export_point_cloud, check_dir

class Single_obj_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Single_obj_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.pre_data()
        if self.mode == "training":
            pass
        elif self.mode == "validation":
            pass
        elif self.mode == "testing":
            pass
        else:
            raise ""

    def pre_data(self):
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return

