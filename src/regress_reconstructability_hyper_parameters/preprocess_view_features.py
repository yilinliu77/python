import math
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import cv2
import hydra
import torch
import torchvision
import numpy as np
import numba as nb
from PIL import Image
from typing import List, Tuple, Optional

from plyfile import PlyData, PlyElement
from torchvision.transforms import transforms
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats
from tqdm.contrib.concurrent import thread_map, process_map

# Output matrix that transform a vector to the local coordinate of the point normal
def calculate_transformation_matrix(v_normal: np.ndarray):
    if len(v_normal.shape)!=2:
        normals = v_normal.reshape((-1,3))
    else:
        normals = v_normal
    up = np.array([0, 0, 1])
    assert np.all(np.abs(np.linalg.norm(normals,axis=1) - 1) < 1e-2)
    z_unit = normals + 1e-6
    z_unit = z_unit / np.linalg.norm(z_unit,axis=1,keepdims=True)
    x_unit = np.cross(z_unit, up)
    x_unit = x_unit / np.linalg.norm(x_unit,axis=1,keepdims=True)
    y_unit = np.cross(z_unit, x_unit)
    y_unit = y_unit / np.linalg.norm(y_unit,axis=1,keepdims=True)
    magic_matrix = np.stack([x_unit, y_unit, z_unit], axis=2)
    assert np.all(np.abs(np.linalg.det(magic_matrix) - 1) < 1e-2)
    magic_matrix = np.transpose(magic_matrix,axes=(0,2,1))

    if len(v_normal.shape)!=2:
        magic_matrix = magic_matrix.reshape(v_normal.shape[:-1]+(3,3))

    return magic_matrix

def compute_view_features(v_max_num_view: int,
                          v_point_feature_root_dir: str,
                          v_img_dir: Optional[str] = None,
                          v_file_content: Optional[Tuple[int, int, List[str]]] = None,
                          ):
    (real_index, num_view, raw_data), v_point_normal = v_file_content
    # raw_data = [item.strip() for item in open(v_file).readlines()]
    num_views = int(num_view)
    if num_views > v_max_num_view - 1:
        raise
    if num_views <= 2:
        return -1., "", False, np.zeros((v_max_num_view, 8), dtype=np.float32), []

    magic_matrix = calculate_transformation_matrix(v_point_normal)

    reconstructability = float(raw_data[1])
    point_feature_path = (os.path.join(v_point_feature_root_dir, str(real_index) + ".npy"))
    valid_flag = True
    img_paths = []
    # Read views
    views = np.zeros((v_max_num_view, 8), dtype=np.float32)
    for i_view in range(num_views):
        view_data = raw_data[2 + i_view].split(",")
        if v_img_dir is not None:
            img_name = view_data[0]
            img_path = os.path.join(v_img_dir, img_name + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(v_img_dir, img_name + ".jpg")
            if not os.path.exists(img_path):
                raise
            img_paths.append(img_path)

        view_to_point = np.array(view_data[1:4], dtype=np.float32)
        distance_ratio = float(view_data[4])
        angle_to_normal_ratio = float(view_data[5])
        angle_to_direction_ratio = float(view_data[6])
        pixel_pos_x = float(view_data[7])
        pixel_pos_y = float(view_data[8])
        views[i_view][0] = 1

        assert np.isclose(np.linalg.norm(view_to_point)/60, distance_ratio, atol = 1e-2)
        point_to_view_normalized = -view_to_point / np.linalg.norm(view_to_point)

        local_view = np.matmul(magic_matrix, point_to_view_normalized)
        local_view = local_view / np.linalg.norm(local_view)
        theta = math.acos(local_view[2])
        phi = math.atan2(local_view[1], local_view[0])
        assert np.isclose(np.dot(point_to_view_normalized, v_point_normal), np.dot(local_view, np.array([0,0,1])), atol = 1e-2)

        views[i_view][1] = theta
        views[i_view][2] = phi
        views[i_view][3] = distance_ratio
        views[i_view][4] = angle_to_normal_ratio
        views[i_view][5] = angle_to_direction_ratio
        views[i_view][6] = pixel_pos_x
        views[i_view][7] = pixel_pos_y

        """
        debug
        """
        # if cv2.getWindowProperty('1', cv2.WND_PROP_VISIBLE) < 1:
        #     cv2.namedWindow("1", cv2.WINDOW_AUTOSIZE)
        # img = cv2.resize(cv2.imread(img_paths[i_view]),(600,400))
        # pt = np.array([pixel_pos_x*600,pixel_pos_y*400],dtype=np.int32)
        # img[
        # np.clip(pt[1]-5,0,400):np.clip(pt[1]+5,0,400),
        # np.clip(pt[0]-5,0,600):np.clip(pt[0]+5,0,600)
        # ]=(0,0,255)
        # cv2.imshow("1", img)
        # cv2.waitKey()
        """
        debug done
        """

        continue

    return reconstructability, point_feature_path, valid_flag, views, img_paths


def preprocess_data(v_root: str, v_error_point_cloud: str, v_img_dir: Optional[str] = None):
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
        if v_img_dir is None:
            avg_recon_error_list = plydata['vertex']['avg_recon_error'].copy()
            avg_gt_error_list = plydata['vertex']['avg_gt_error'].copy()
        else:
            avg_recon_error_list=np.zeros_like(x)
            avg_gt_error_list=np.zeros_like(y)
        x_dim = (np.max(x) - np.min(x)) / 2
        y_dim = (np.max(y) - np.min(y)) / 2
        z_dim = (np.max(z) - np.min(z)) / 2
        max_dim = max(x_dim, y_dim, z_dim)
        np.savez(os.path.join(v_root, "../data_centralize"), np.array([np.mean(x), np.mean(y), np.mean(z), max_dim]))
        x = (x - np.mean(x)) / max_dim
        y = (y - np.mean(y)) / max_dim
        z = (z - np.mean(z)) / max_dim
        error_list = np.stack([avg_recon_error_list, avg_gt_error_list, x, y, z, avg_recon_error_list < 0, nx, ny, nz],
                              axis=1)
    point_feature_root_dir=""
    if v_img_dir is None:
        point_feature_root_dir = open(os.path.join(v_root, "../img_dataset_path.txt")).readline()

    num_points = error_list.shape[0]  # This is the number of the total sample points

    # Read the reconstructability file
    # The total number might not be equal to the $num_points$, because some points can not be seen by the given views
    def read_reconstructability_file(v_root: str, v_id: int) -> Tuple[int, int, List[str]]:
        file_name = "{}.txt".format(v_id)
        file_name = os.path.join(v_root, file_name)
        if not os.path.exists(file_name):
            return (v_id, -1, [""])
        raw_data = [item.strip() for item in open(file_name).readlines()]
        num_views = int(raw_data[0])
        return (v_id, num_views, raw_data)

    data_content = thread_map(partial(read_reconstructability_file, v_root), range(num_points), max_workers=4)
    max_num_view = max(data_content, key=lambda x: x[1])[1] + 1
    print("Read attribute with max view: ", max_num_view)

    # Collect views
    cur = time.time()
    views_result = process_map(partial(compute_view_features,
                                      max_num_view, point_feature_root_dir, v_img_dir),
                              zip(data_content, error_list[:, 6:9]), max_workers=10,chunksize=4096)
    print(time.time()-cur)
    reconstructability = np.asarray(list(map(lambda x: x[0], views_result)))
    point_features_path = np.asarray(list(map(lambda x: x[1], views_result)))
    valid_flag = np.asarray(list(map(lambda x: x[2], views_result)))
    views = np.asarray(list(map(lambda x: x[3], views_result)))
    img_paths = np.asarray(list(map(lambda x: x[4], views_result)))

    # views_pair = np.zeros((num_points, max_num_view, max_num_view, 3), dtype=np.float16)
    views_pair = None

    views = views[valid_flag]
    reconstructability = reconstructability[valid_flag]
    point_features_path = point_features_path[valid_flag]
    img_paths = img_paths[valid_flag]
    # views_pair = views_pair[valid_flag]
    # usable_indices = np.triu_indices(max_num_view, 1)
    # views_pair = views_pair[:, usable_indices[0], usable_indices[1]]

    error_list = error_list[valid_flag]
    point_attribute = np.concatenate([reconstructability[:, np.newaxis], error_list], axis=1).astype(np.float32)
    print(
        "Totally {} points; {} points has error 0; {} points has 0 reconstructability; Output {} points".format(
            num_points,
            (point_attribute[:, 1] == -1).sum(),
            num_points - valid_flag.sum(),
            point_attribute.shape[0]
        ))
    return views, views_pair, point_attribute, point_features_path, img_paths


if __name__ == '__main__':
    import sys

    v_data_root = sys.argv[1]
    v_output_root = sys.argv[2]
    reconstructability_file_dir = os.path.join(v_data_root, "reconstructability")
    error_point_cloud_dir = os.path.join(v_data_root, "accuracy_projected.ply")
    img_rescale_size = (400, 600)

    if not os.path.exists(v_output_root):
        os.mkdir(v_output_root)
    view, view_pair, point_attribute, point_features_path, img_paths = preprocess_data(
        reconstructability_file_dir,
        error_point_cloud_dir
    )
    np.save(os.path.join(v_output_root, "views"), view)
    # np.savez_compressed(os.path.join(output_root, "training_data/view_pairs"), view_pair)
    np.save(os.path.join(v_output_root, "point_attribute"), point_attribute)
    np.save(os.path.join(v_output_root, "view_paths"), point_features_path)
    np.save(os.path.join(v_output_root, "img_paths"), img_paths)
    print("Pre-compute data done")
