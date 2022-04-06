import math
import os
import pickle
from functools import partial
from multiprocessing import Pool

import cv2
import hydra
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import List, Tuple

from plyfile import PlyData, PlyElement
from torchvision.transforms import transforms
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats
from tqdm.contrib.concurrent import thread_map, process_map


def compute_view_features(max_num_view, valid_views_flag, reconstructabilities,
                          views, views_pair, point_feature_paths,point_feature_root_dir, error_list,
                          v_file):
    real_index = int(v_file.split("\\")[-1][:-4])

    raw_data = [item.strip() for item in open(v_file).readlines()]
    num_views = int(raw_data[0])
    if num_views > max_num_view - 1:
        raise
    reconstructability = float(raw_data[1])
    if num_views >=2: # Unstable change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if reconstructability != 0:
        point_feature_paths.append(os.path.join(point_feature_root_dir, str(real_index) + ".npz"))
        valid_views_flag[real_index] = True
    else:
        print(0)
    reconstructabilities[real_index] = reconstructability
    # Read views
    for i_view in range(num_views):
        view_data = raw_data[2 + i_view].split(",")
        img_name = view_data[0]
        img_name = os.path.join(point_feature_root_dir, "../../images/{}.png".format(img_name))
        view_to_point = np.array(view_data[1:4], dtype=np.float32)
        distance_ratio = float(view_data[4])
        angle_to_normal_ratio = float(view_data[5])
        angle_to_direction_ratio = float(view_data[6])
        pixel_pos_x = float(view_data[7])
        pixel_pos_y = float(view_data[8])
        views[real_index][i_view][0] = 1

        assert abs(np.linalg.norm(view_to_point) - distance_ratio * 60) < 1
        point_to_view_normalized = -view_to_point / np.linalg.norm(view_to_point)
        point_to_view_theta = math.acos(point_to_view_normalized[2])
        point_to_view_phi = math.atan2(point_to_view_normalized[1], point_to_view_normalized[0])

        normal_normalized = error_list[real_index][5:8]
        normal_theta = math.acos(normal_normalized[2])
        normal_phi = math.atan2(normal_normalized[1], normal_normalized[0])

        delta_theta:float = point_to_view_theta - normal_theta
        delta_phi:float = point_to_view_phi - normal_phi
        while delta_theta < -math.pi:
            delta_theta+=math.pi*2
        while delta_theta > math.pi:
            delta_theta-=math.pi*2
        while delta_phi < -math.pi:
            delta_phi += math.pi * 2
        while delta_phi > math.pi:
            delta_phi -= math.pi * 2

        views[real_index][i_view][1] = delta_theta
        views[real_index][i_view][2] = delta_phi
        views[real_index][i_view][3] = distance_ratio
        views[real_index][i_view][4] = angle_to_normal_ratio
        views[real_index][i_view][5] = angle_to_direction_ratio
        views[real_index][i_view][6] = pixel_pos_x
        views[real_index][i_view][7] = pixel_pos_y
        continue

    # Read view pair
    # cur_iter = 0
    # for i_view1 in range(max_num_view):
    #     for i_view2 in range(i_view1 + 1, max_num_view):
    #         if i_view1 >= num_views or i_view2 >= num_views:
    #             continue
    #         view_pair_data = raw_data[2 + num_views + cur_iter].split(",")
    #         alpha_ratio = float(view_pair_data[0])
    #         relative_distance_ratio = float(view_pair_data[1])
    #         views_pair[real_index][i_view1][i_view2][0] = 1
    #         views_pair[real_index][i_view1][i_view2][1] = alpha_ratio
    #         views_pair[real_index][i_view1][i_view2][2] = relative_distance_ratio
    #         cur_iter += 1

def preprocess_data(v_root: str, v_error_point_cloud: str) -> (np.ndarray, np.ndarray):
    print("Read point cloud")
    with open(v_error_point_cloud, "rb") as f:
        plydata = PlyData.read(f)
        x = plydata['vertex']['x'].copy()
        y = plydata['vertex']['y'].copy()
        z = plydata['vertex']['z'].copy()
        nx = plydata['vertex']['nx'].copy()
        ny = plydata['vertex']['ny'].copy()
        nz = plydata['vertex']['nz'].copy()
        length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        nx = nx / length
        ny = ny / length
        nz = nz / length
        avg_recon_error_list = plydata['vertex']['avg_recon_error'].copy()
        avg_gt_error_list = plydata['vertex']['avg_gt_error'].copy()
        x_dim = (np.max(x) - np.min(x)) / 2
        y_dim = (np.max(y) - np.min(y)) / 2
        z_dim = (np.max(z) - np.min(z)) / 2
        max_dim = max(x_dim, y_dim, z_dim)
        np.savez(os.path.join(v_root,"../data_centralize"),np.array([np.mean(x),np.mean(y),np.mean(z),max_dim]))
        x = (x - np.mean(x)) / max_dim
        y = (y - np.mean(y)) / max_dim
        z = (z - np.mean(z)) / max_dim
        error_list = np.stack([avg_recon_error_list, avg_gt_error_list, x, y, z, avg_recon_error_list < 0, nx, ny, nz], axis=1)

    point_feature_root_dir = open(os.path.join(v_root,"../img_dataset_path.txt")).readline()
    # point_feature_root_dir = None

    files = [os.path.join(v_root, item) for item in os.listdir(v_root)]
    files = list(filter(lambda item: ".txt" in item, files))
    files = sorted(files, key=lambda item: int(item.split("\\")[-1][:-4]))

    num_points = error_list.shape[0]
    # Find max view numbers
    def compute_max_view_number(v_file):
        raw_data = [item.strip() for item in open(v_file).readlines()]
        num_views = int(raw_data[0])
        return num_views
    num_view_list = thread_map(compute_max_view_number, files, max_workers=8)
    max_num_view = max(num_view_list) + 1
    # max_num_view = 200
    print("Read attribute with max view: ", max_num_view)
    point_features_path = []
    views = np.zeros((num_points, max_num_view, 8), dtype=np.float16)
    # views_pair = np.zeros((num_points, max_num_view, max_num_view, 3), dtype=np.float16)
    views_pair = None
    valid_views_flag = [False for _ in range(num_points)]
    reconstructabilities = [0 for _ in range(num_points)]

    thread_map(partial(compute_view_features,
                       max_num_view,valid_views_flag,reconstructabilities,views,views_pair,
                       point_features_path,point_feature_root_dir,error_list),
               files,max_workers=10)

    # valid_flag = np.logical_and(np.array(valid_views_flag), error_list[:, 2] != 0)
    valid_flag = np.array(valid_views_flag)

    views = views[valid_flag]
    # views_pair = views_pair[valid_flag]
    point_features_path = np.array(point_features_path)
    reconstructabilities = np.asarray(reconstructabilities)[valid_flag]
    usable_indices = np.triu_indices(max_num_view, 1)
    # views_pair = views_pair[:, usable_indices[0], usable_indices[1]]

    error_list = error_list[valid_flag]
    point_attribute = np.concatenate([reconstructabilities[:, np.newaxis], error_list], axis=1).astype(np.float16)
    print("Totally {} points; {} points has error 0; {} points has 0 reconstructability, {} in it is not visible; Output {} points".format(
        num_points,
        error_list[:, -1].sum(),
        num_points-valid_flag.sum(),
        num_points-len(files),
        point_attribute.shape[0]
    ))
    return views, views_pair, point_attribute, point_features_path


if __name__ == '__main__':
    import sys

    output_root = sys.argv[1]
    reconstructability_file_dir = os.path.join(output_root, "reconstructability")
    error_point_cloud_dir = os.path.join(output_root, "accuracy_projected.ply")
    img_rescale_size = (400, 600)

    if not os.path.exists(os.path.join(output_root, "training_data")):
        os.mkdir(os.path.join(output_root, "training_data"))
    view, view_pair, point_attribute, view_paths = preprocess_data(
        reconstructability_file_dir,
        error_point_cloud_dir
    )
    np.savez_compressed(os.path.join(output_root, "training_data/views"), view)
    # np.savez_compressed(os.path.join(output_root, "training_data/view_pairs"), view_pair)
    np.savez_compressed(os.path.join(output_root, "training_data/point_attribute"), point_attribute)
    np.savez_compressed(os.path.join(output_root, "training_data/view_paths"), view_paths)
    print("Pre-compute data done")
