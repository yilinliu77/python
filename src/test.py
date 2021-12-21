import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pymap3d as pm
from pyproj import Transformer,CRS
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


def test1():
    num_item = 100000
    a = np.random.random((num_item,200))
    if os.path.exists("test1"):
        shutil.rmtree("test1")
    os.mkdir("test1")
    print("Save")
    def test_thread(v_id):
        read_index = np.random.randint(0, num_item)
        read_filename = "test1/{}".format(read_index)
        if os.path.exists(read_filename):
            np.load(read_filename)
        else:
            for i in range(max(0,read_index-10),min(num_item,read_index+10)):
                save_filename = "test1/{}".format(i)
                np.savez(save_filename, a[i])
    # for i in tqdm(range(num_item)):
    #     test_thread(i)

    r = thread_map(test_thread, range(num_item), max_workers=4)

    return


if __name__ == '__main__':
    test1()
    # plt.figure()
    # plt.subplot(2,1,1)
    # x = np.random.randint(0,500,100)
    # y = np.random.standard_normal(100)
    # plt.scatter(x,y,color="g")
    # plt.xlabel("Random integers")
    # plt.ylabel("Random normal dist.")
    # plt.title("A beautiful Scatter Plot of the Standard normal distribution")
    #
    # plt.subplot(2, 1, 2)
    # x = np.arange(0,11,0.01)
    # y = np.sin(x)
    # plt.plot(x, y)
    # plt.xlabel("x")
    # plt.ylabel("sin(x)")
    # plt.title("Visualizing Sin")
    #
    # plt.tight_layout()
    # plt.show()
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3857))
    wgs_long_lat=np.array([113.934737085972,22.5322469005556,121.757])
    print(transformer.transform(wgs_long_lat[1],wgs_long_lat[0]))

    transformer = Transformer.from_crs(CRS.from_epsg(4547), CRS.from_epsg(3857))
    cgcs2000_4547=np.array([-243.017,-370.281,-127.169])+np.array([493487.4722,2493097.072,179])

    print(transformer.transform(cgcs2000_4547[1],cgcs2000_4547[0]))