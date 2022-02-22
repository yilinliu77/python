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
    # ready to run example: PythonClient/multirotor/hello_drone.py
    import airsim
    import os

    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Async methods returns Future. Call join() to wait for task to complete.
    client.takeoffAsync().join()
    client.moveToPositionAsync(-10, 10, -10, 5).join()

    # take images
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
    print('Retrieved images: %d', len(responses))

    # do something with the images
    for response in responses:
        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm(os.path.normpath('test/py1.pfm'), airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            airsim.write_file(os.path.normpath('test/py1.png'), response.image_data_uint8)