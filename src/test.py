import os
import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pymap3d as pm
from plyfile import PlyData
from pyproj import Transformer, CRS
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

if __name__ == '__main__':
    with open(r"C:\repo\C\build\src\sig22_reconstructability\optimize_trajectory\log\init_points.ply", "rb") as f:
        plydata = PlyData.read(f)
        c_recon = plydata['vertex']['recon'].copy()

        filename = list(
            map(lambda item: int(item[:-4]),
                filter(
                    lambda x: x[-4:] == ".txt",
                    os.listdir(r"D:\Projects\Reconstructability\PathPlanning\test_predict\reconstructability")))
        )
        filename.sort()
        c_recon=c_recon[filename]
    python_recon = np.zeros_like(c_recon)
    with open(r"C:\repo\python\temp\test_scene_output\whole_point.ply", "rb") as f:
        plydata = PlyData.read(f)
        part_python_recon = plydata['vertex']['Predict_Recon'].copy()

    pass