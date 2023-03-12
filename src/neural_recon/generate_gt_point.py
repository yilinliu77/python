import math
from typing import Tuple

import numpy as np
import open3d as o3d
import os

import ray
from numba import prange
from scipy.spatial.distance import cdist
from tqdm import tqdm
import cv2
import numba as nb

import sys


@nb.njit()
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if abs(stds) < 1e-6:
        return 0.
    else:
        product /= stds
        return product


@nb.njit()
def calculate_ncc_batch(v_roi_regions):
    num_region = v_roi_regions.shape[0]
    nccs = -np.ones(int(num_region * num_region), np.float32)
    for id_view1 in range(num_region):
        for id_view2 in range(id_view1 + 1, num_region):
            ncc = correlation_coefficient(v_roi_regions[id_view1].astype(np.float32),
                                          v_roi_regions[id_view2].astype(np.float32))
            nccs[id_view1 * num_region + id_view2] = ncc
    nccs_ = nccs[nccs != -1]
    return nccs_


# @nb.jit(forceobj=True)
def compute_loss_item(v_imgs, v_num_view, projected_points, img_names):
    is_debug=False
    # is_debug=True
    half_window_sizes = [3, 7, 14]
    window_sizes = [item * 2 + 1 for item in half_window_sizes]
    roi_regions = [[] for _ in window_sizes]
    # It might not equal to `num_view`. Some projections might be a single pixel on image. Just skip them
    for id_view in range(v_num_view):
        # Take the data
        img = v_imgs[img_names[id_view]]
        projected_point = projected_points[id_view]
        img_size = img.shape[:2][::-1]
        ori_coords = (projected_point * img_size).astype(np.int64)

        if is_debug:
            viz_img = cv2.circle(img.copy(), ori_coords, 5, (255, 0, 0), 2)
            cv2.imwrite("output/loss_test/0_{}_original.png".format(id_view), viz_img)
        # Compute the related region using a rotated rectangle
        if ori_coords[0] < max(half_window_sizes) \
                or ori_coords[1] < max(half_window_sizes) \
                or ori_coords[0] >= img_size[0] - max(half_window_sizes) \
                or ori_coords[1] >= img_size[1] - max(half_window_sizes):
            continue

        # ROI region
        roi_resized = [
            img[
            ori_coords[1] - half_window_size:ori_coords[1] + half_window_size + 1,
            ori_coords[0] - half_window_size:ori_coords[0] + half_window_size + 1
            ]
            for half_window_size in half_window_sizes]
        for idx, _ in enumerate(window_sizes):
            roi_regions[idx].append(roi_resized[idx])
            if is_debug:
                cv2.imwrite("output/loss_test/0_{}_cropped_{}.png".format(id_view, idx), roi_resized[idx])

    if len(roi_regions[0]) <= 1:
        final_ncc = 0.
    else:
        nccs = []
        for idx, _ in enumerate(window_sizes):
            nccs.append(calculate_ncc_batch(np.stack(roi_regions[idx], axis=0)))  # [-1,1]
        final_ncc = np.mean(nccs) / 2 + 0.5  # [0,1]
        pass
    return final_ncc


@ray.remote
def compute_loss(
        v_id_worker,
        v_num_sample_per_worker,
        v_points,
        v_points_visibilities,
        v_projections, v_img_names, v_imgs,
        progress_actor: ray.actor.ActorHandle,
        v_is_dummy: int
):
    start_index = v_id_worker * v_num_sample_per_worker
    end_index = (v_id_worker + 1) * v_num_sample_per_worker
    if end_index > v_points.shape[0]:
        end_index = v_points.shape[0]
    real_num = end_index - start_index
    gt_loss = np.zeros((real_num, 1), dtype=np.float16)
    projected_points = [[] for _ in range(real_num)]
    if v_is_dummy != -1:
        v_num_sample_per_worker = 1
    for id_local in range(v_num_sample_per_worker):
        id_global = v_id_worker * v_num_sample_per_worker + id_local
        if id_global >= v_points.shape[0]:
            break
        if v_is_dummy != -1:
            id_global = v_is_dummy
        point = v_points[id_global].astype(np.float32)
        visibility_mask = v_points_visibilities[:, id_global]

        img_names = [item for idx, item in enumerate(v_img_names) if visibility_mask[idx]]
        projection_matrix_ = [item for idx, item in enumerate(v_projections) if visibility_mask[idx]]

        # Project the segments on image
        homogeneous_sample_point = np.insert(point, 3, 1)
        num_view = len(img_names)
        # We should at least have two views to reconstruct a target
        # or we will predict it as a negative sample
        if num_view <= 1:
            projected_point = np.zeros((0, 2))
            final_ncc = 0.
        else:
            # Calculate the projected segment
            projected_point = np.matmul(np.asarray(projection_matrix_),
                                         homogeneous_sample_point)
            projected_point = projected_point[:, :2] / projected_point[:, 2:3]

            final_ncc = compute_loss_item(v_imgs,
                                          num_view,
                                          projected_point,
                                          img_names)
        gt_loss[id_local, 0] = final_ncc
        projected_points[id_local] = projected_point
        if v_is_dummy == -1 and v_num_sample_per_worker // 100 > 0 and id_local % (v_num_sample_per_worker // 100) == 0:
            progress_actor.report_progress.remote(v_id_worker, id_local)

    if v_is_dummy == -1:
        os.makedirs("output/gt_loss/sub/", exist_ok=True)
        np.save("output/gt_loss/sub/projected_points_{}".format(v_id_worker),
                np.array(projected_points, dtype=object))
        np.save("output/gt_loss/sub/gt_loss_{}".format(v_id_worker), gt_loss)
        progress_actor.report_progress.remote(v_id_worker, v_num_sample_per_worker)
    return


@ray.remote
class ProgressActor:
    def __init__(self, total_num_samples: int):
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task = {}

    def report_progress(self, task_id: int, num_finished) -> None:
        self.num_samples_completed_per_task[task_id] = num_finished

    def get_progress(self) -> float:
        return (
                sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
        )
