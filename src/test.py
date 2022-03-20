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

import sympy

def bak():
    data = ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29", "31"]
    for his_month in tqdm(data):
        for his_day in data:
            for her_month in data:
                for her_day in data:
                    flag = True

                    sum_his = 0
                    sum_her = 0
                    for ch in his_month:
                        sum_his += int(ch)
                    for ch in his_day:
                        sum_his += int(ch)
                    for ch in her_month:
                        sum_her += int(ch)
                    for ch in her_day:
                        sum_her += int(ch)
                    if not sympy.isprime(sum_his):  # 3
                        continue
                    if not sympy.isprime(sum_her):  # 3
                        continue
                    if not sympy.isprime(int(her_month + her_day)):  # 4
                        continue
                    if sympy.isprime(int((her_month + her_day)[::-1])):  # 4
                        continue
                    if not sympy.isprime(int(his_month + his_day)):  # 5
                        continue
                    if sympy.isprime(int((his_month + his_day)[::-1])):  # 5
                        continue
                    if not sympy.isprime(int((his_day + her_month))):  # 6
                        continue
                    if sympy.isprime(int((his_day + her_month)[::-1])):  # 6
                        continue
                    if not sympy.isprime(int((his_month + her_day))):  # 7
                        continue
                    if sympy.isprime(int((his_month + her_day)[::-1])):  # 7
                        continue
                    if sympy.isprime(int((her_day + his_month))):  # 8
                        continue
                    if not sympy.isprime(int((her_day + his_month)[::-1])):  # 8
                        continue
                    if sympy.isprime(int((her_month + his_day))):  # 9
                        continue
                    if sympy.isprime(int((her_month + his_day)[::-1])):  # 9
                        continue
                    if not sympy.isprime(sum_his - sum_her):  # 10
                        # if not sympy.isprime(abs(sum_her-sum_his)):  # 10
                        continue

                    print(his_month)
                    print(his_day)
                    print(her_month)
                    print(her_day)

# if __name__ == '__main__':
#
#     with open(r"D:\Projects\Reconstructability\training_data\v2\xiaozhen_fine_ds_80", "rb") as f:
#         plydata = PlyData.read(f)
#         c_recon = plydata['vertex']['recon'].copy()
#
#         filename = list(
#             map(lambda item: int(item[:-4]),
#                 filter(
#                     lambda x: x[-4:] == ".txt",
#                     os.listdir(r"D:\Projects\Reconstructability\PathPlanning\test_predict\reconstructability")))
#         )
#         filename.sort()
#         c_recon = c_recon[filename]
#     python_recon = np.zeros_like(c_recon)
#     with open(r"C:\repo\python\temp\test_scene_output\whole_point.ply", "rb") as f:
#         plydata = PlyData.read(f)
#         part_python_recon = plydata['vertex']['Predict_Recon'].copy()
#
#     pass

if __name__ == '__main__':
    dirs = os.listdir(r"D:\Projects\Reconstructability\training_data\v4")
    dirs.remove("raw")
    info_matrix = np.zeros((len(dirs),8)).astype(np.object)
    for id, dir in enumerate(tqdm(dirs)):
        views = np.load(os.path.join(r"D:\Projects\Reconstructability\training_data\v4",dir,"training_data/views.npz"))["arr_0"].astype(np.float32)
        points = np.load(os.path.join(r"D:\Projects\Reconstructability\training_data\v4",dir,"training_data/point_attribute.npz"))["arr_0"].astype(np.float32)
        valid_error = points[points[:,5]==0,1]
        # print(dir)
        # print(views.shape[0])
        # print(views.shape[1])
        # print(views[:,1,0].sum())
        # print(points.shape[0]-points[:,5].sum())
        # print(valid_error.mean())
        # print(np.percentile(valid_error,50))
        # print(np.percentile(valid_error,90))
        # print("====================================================")
        # print("====================================================")
        info_matrix[id,0] = dir
        info_matrix[id,1] = views.shape[0]
        info_matrix[id,2] = views.shape[1]
        info_matrix[id,3] = views[:,1,0].sum()
        info_matrix[id,4] = points.shape[0]-points[:,5].sum()
        info_matrix[id,5] = valid_error.mean()
        info_matrix[id,6] = np.percentile(valid_error,50)
        info_matrix[id,7] = np.percentile(valid_error,90)
        # break

    np.savetxt("temp/info.csv",info_matrix, fmt="%s", delimiter=',')