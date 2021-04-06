import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch

from scipy.spatial import Delaunay
import numpy as np



if __name__ == '__main__':
    pcd=o3d.io.read_point_cloud("temp/bunny_2d.ply")
    points=np.asarray(pcd.points)

