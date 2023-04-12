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
from shared.common_utils import to_homogeneous, save_line_cloud, normalize_vector, normalized_torch_img_to_numpy, \
    to_homogeneous_tensor
from src.neural_recon.optimize_segment import optimize_single_segment, optimize_single_segment_tensor
from src.neural_recon.phase1 import Phase1
import faiss

from src.neural_recon.phase3 import prepare_dataset_and_model, LModel17, LModel18

if __name__ == '__main__':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    data = prepare_dataset_and_model("d:/Projects/NeuralRecon/Test_data/OBL_L7/Test_imgs2_colmap_neural/sparse_align",
                                     "output/neural_recon/ngp_models",
                                     9118)

    model = LModel18(data, [1,0,0],
             "sample",
             "outputs/testviz")

    point_index = model.batched_points_per_patch[1476]
    point_index = point_index[3*4:4*4]
    ray_c = model.ray_c[point_index].reshape((-1, 4, 3))

    for idx, distance in enumerate(torch.arange(70,160,1,device=ray_c.device)):
        model.zero_grad()
        model.seg_distance.data[point_index[0]] = distance / model.seg_distance_normalizer
        seg_distance = model.seg_distance[point_index].reshape((-1, 4, 1)) * model.seg_distance_normalizer

        edge_points = ray_c * seg_distance
        edge_points[:,1:] = edge_points[:,1:].detach()

        up_c = model.get_up_vector([point_index], edge_points[:, 0], edge_points[:, 1])
        _, coords_per_edge = model.sample_points_based_on_up(edge_points[:, 0], edge_points[:, 1], up_c[:,0])
        loss = model.compute_similarity(coords_per_edge, model.intrinsic1, model.transformation)

        loss.backward()
        print("{}: {:.6f}, {:.6f}".format(distance, loss, model.seg_distance.grad[point_index[0]]))

        line_thickness = 1
        point_thickness = 2
        point_radius = 1

        polygon_points_2d_1 = (model.intrinsic1 @ coords_per_edge.T).T
        polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
        polygon_points_2d_2 = (model.transformation @ to_homogeneous_tensor(coords_per_edge).T).T
        polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()

        line_img1 = model.rgb1.copy()
        line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
        shape = line_img1.shape[:2][::-1]

        roi_coor_2d1_numpy = np.clip(polygon_points_2d_1, 0, 0.99999)
        viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
        line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

        polygon_2d1 = (model.intrinsic1 @ edge_points[0].T).T
        polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
        polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
        cv2.line(line_img1, polygon_2d1[0], polygon_2d1[1],
                 color=(0, 255, 0), thickness=line_thickness)
        cv2.circle(line_img1, polygon_2d1[0], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)
        cv2.circle(line_img1, polygon_2d1[1], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

        # Image 2
        line_img2 = model.rgb2.copy()
        line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
        shape = line_img2.shape[:2][::-1]

        roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
        viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
        line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

        polygon_2d2 = (model.transformation @ to_homogeneous_tensor(edge_points[0]).T).T
        polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
        polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
        cv2.line(line_img2, polygon_2d2[0], polygon_2d2[1],
                 color=(0, 255, 0), thickness=line_thickness)
        cv2.circle(line_img2, polygon_2d2[0], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)
        cv2.circle(line_img2, polygon_2d2[1], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

        cv2.imwrite(os.path.join(model.log_root, "{:05d}.jpg".format(idx)),
                    np.concatenate((line_img1, line_img2), axis=0))

    pass