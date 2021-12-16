import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pymap3d as pm
from pyproj import Transformer,CRS

if __name__ == '__main__':
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