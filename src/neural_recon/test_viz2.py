import itertools
import math
import random
import sys, os

import open3d

from shared.perspective_geometry import extract_fundamental_from_projection
from src.neural_recon.colmap_io import read_dataset, Image, Point_3d

import cv2
import numpy as np
import torch
from tqdm import tqdm
from shared.common_utils import to_homogeneous, save_line_cloud, normalize_vector, normalized_torch_img_to_numpy
from src.neural_recon.optimize_segment import optimize_single_segment, optimize_single_segment_tensor
from src.neural_recon.phase1 import Phase1
import faiss

from src.neural_recon.phase3 import prepare_dataset_and_model, LModel17

if __name__ == '__main__':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    data = prepare_dataset_and_model("d:/Projects/NeuralRecon/Test_data/OBL_L7/Test_imgs2_colmap_neural/sparse_align",
                                     119)

    model = LModel17(data, [1,0,0],
             "model",
             "outputs/testviz")

    point_index = model.edge_point_index[119]
    point_index = point_index[7*4:8*4]
    ray_c = model.ray_c[point_index].reshape((-1, 4, 3))
    seg_distance = model.seg_distance[point_index].reshape((-1, 4, 1)) * model.seg_distance_normalizer
    point_pos_c = ray_c * seg_distance
    edge_points = point_pos_c

    losses=[]
    for id_x, t in enumerate(torch.arange(0,1,0.01)):
        model.v_up.data = torch.ones_like(model.v_up.data) * t
        up_c = model.get_up_vector2([119], edge_points[:, 0], edge_points[:, 1])
        up_c = up_c[7:8]

        loss = model.sample_points_based_on_up(edge_points, up_c, id_x, True)
        losses.append(loss[0].item())
        pass
    pass