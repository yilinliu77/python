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


# The similarity is between 0 and 1
# The magnitude is between 0 and 1
# Larger is better
# v_segment: (2, 2)
# v_direction_img: (h, w, 2)
@nb.njit()
def compute_direction_similarity(v_segment: np.ndarray, v_direction_img: np.ndarray):
    img_size = np.asarray(v_direction_img.shape[:2][::-1], dtype=np.float32)
    source_direction = v_segment[1] - v_segment[0]
    source_norm = np.linalg.norm(source_direction)

    # Compute the sampled points along the edge
    delta = np.expand_dims(source_direction / 10, 0)
    sample_points = np.dot(np.expand_dims(np.asarray(np.arange(11), dtype=np.float32), 1), delta)
    sample_points += v_segment[0:1]
    sample_point_coordinate = (sample_points * img_size).astype(np.int64)

    # Sample
    sample_direction = np.zeros((sample_point_coordinate.shape[0], 2), np.float32)
    for n, (y, x) in enumerate(zip(sample_point_coordinate[:, 1], sample_point_coordinate[:, 0])):
        sample_direction[n] = v_direction_img[y, x]
    sample_direction_norm = np.sqrt(sample_direction[:, 0] ** 2 + sample_direction[:, 1] ** 2)

    # Compute angle
    # The direction can be negative. Compute both angle and find the smallest
    angle_cos1 = np.abs((sample_direction * source_direction).sum(axis=1) / (
            1e-8 + sample_direction_norm * source_norm))
    angle_cos2 = np.abs((sample_direction * -source_direction).sum(axis=1) / (
            1e-8 + sample_direction_norm * source_norm))

    # Find the smallest angle
    angle_cos = np.zeros(angle_cos1.shape[0], np.float32)
    for i in prange(angle_cos1.shape[0]):
        angle_cos[i] = min(angle_cos1[i], angle_cos2[i])
    return 1 - np.mean(angle_cos), np.mean(sample_direction_norm)


# @nb.njit()
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if abs(stds) < 1e-6:
        return 0.
    else:
        product /= stds
        return product


# @nb.njit()
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


@nb.njit()
def calculate_lbd_score_batch(v_descriptors):
    num_region = v_descriptors.shape[0]
    scores = -np.ones(int(num_region * num_region), np.float32)
    for id_view1 in range(num_region):
        for id_view2 in range(id_view1 + 1, num_region):
            score = np.mean(np.sqrt(((v_descriptors[id_view1] - v_descriptors[id_view2]) ** 2)))
            scores[id_view1 * num_region + id_view2] = score
    scores_ = scores[scores != -1]
    return scores_


# Compute the related region using a rotated rectangle
# v_normalized_segment: (2, 2)
# v_img_size: (h, w)
@nb.njit()
def extract_roi_rectangle(v_normalized_segment: np.ndarray, v_img_size: Tuple):
    img_size_ = np.asarray(v_img_size, np.float32)
    projected_segment_per_view_original = v_normalized_segment * img_size_

    segment_direction = projected_segment_per_view_original[1, :] - projected_segment_per_view_original[0, :]
    segment_length = np.sqrt(segment_direction[0] ** 2 + segment_direction[1] ** 2)

    roi_size = np.asarray((segment_length, 10.), np.int64)
    roi_angle = np.arctan2(segment_direction[1], segment_direction[0])
    roi_center = (projected_segment_per_view_original.sum(axis=0)) / 2

    return roi_size, roi_angle, roi_center


# @nb.jit(forceobj=True)
def extract_roi_region(v_img: np.ndarray, v_roi_center, v_roi_angle, v_roi_size, id_view):
    diag_length = np.linalg.norm(v_roi_size)
    aabb_roi = cv2.getRectSubPix(v_img, np.ceil((diag_length, diag_length)).astype(np.int64), v_roi_center)
    M = cv2.getRotationMatrix2D((aabb_roi.shape[1] / 2, aabb_roi.shape[0] / 2), np.rad2deg(v_roi_angle), 1)
    img_rotated = cv2.warpAffine(aabb_roi, M, aabb_roi.shape[:2][::-1], cv2.INTER_CUBIC)
    roi = cv2.getRectSubPix(img_rotated, (v_roi_size[0], v_roi_size[1]), (aabb_roi.shape[1] / 2, aabb_roi.shape[0] / 2))
    roi_resized = cv2.resize(roi, (50, 5), interpolation=cv2.INTER_AREA)
    if True:
        cv2.imwrite("output/loss_test/1_{}_rotated.png".format(id_view), img_rotated)
        cv2.imwrite("output/loss_test/2_{}_roi.png".format(id_view), roi)
        cv2.imwrite("output/loss_test/3_{}_roi_resized.png".format(id_view), roi_resized)
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("1", 1600, 900)
        # cv2.moveWindow("1", 0, 0)
        # cv2.imshow("1", np.concatenate((img_viz, img_rotated), axis=1))
        # cv2.waitKey()
        # cv2.imshow("1", roi_resized)
        # cv2.waitKey()
    return roi_resized


@nb.jit(forceobj=True)
def keyline_from_seg(v_segments):
    keylines = [cv2.line_descriptor.KeyLine() for i in range(v_segments.shape[0])]
    for i in range(v_segments.shape[0]):
        kl = cv2.line_descriptor.KeyLine()
        kl.startPointX = float(v_segments[i][0])
        kl.startPointY = float(v_segments[i][1])
        kl.endPointX = float(v_segments[i][2])
        kl.endPointY = float(v_segments[i][3])
        kl.sPointInOctaveX = float(v_segments[i][0])
        kl.sPointInOctaveY = float(v_segments[i][1])
        kl.ePointInOctaveX = float(v_segments[i][2])
        kl.ePointInOctaveY = float(v_segments[i][3])

        dx = kl.endPointX - kl.startPointX
        dy = kl.endPointY - kl.startPointY
        kl.angle = math.atan2(dy, dx)
        kl.class_id = i
        kl.octave = 0
        kl.pt = (
            0.5 * (kl.endPointX + kl.startPointX),
            0.5 * (kl.endPointY + kl.startPointY)
        )
        kl.size = 0
        kl.lineLength = math.sqrt(dx * dx + dy * dy)
        kl.numOfPixels = int(max(abs(dx), abs(dy)))

        kl.response = 0
        keylines[i] = kl
    return keylines


# @nb.jit(forceobj=True)
def compute_loss_item(v_imgs, v_num_view, projected_segment, img_names):
    # Compute the related region (resize to (W*H)) on the image according to the projected segment
    # W is the length of the projected segment
    # H is a pre-defined parameter (10 by default)
    roi_regions = []
    edge_similarities = []
    edge_magnitudes = []
    lbd_descriptors = []
    # It might not equal to `num_view`. Some projections might be a single pixel on image. Just skip them
    for id_view in range(v_num_view):
        # Take the data
        img = v_imgs[img_names[id_view]][0]
        projected_segment_per_view = projected_segment[id_view]
        img_size = img.shape[:2][::-1]
        ori_coords = (projected_segment_per_view * img_size).astype(np.int64)

        if True:
            viz_img = cv2.line(img.copy(), ori_coords[0], ori_coords[1], (255, 0, 0), 2)
            cv2.imwrite("output/loss_test/0_{}_original.png".format(id_view), viz_img)
        # Compute the related region using a rotated rectangle
        roi_size, roi_angle, roi_center = extract_roi_rectangle(projected_segment_per_view, img_size)
        # Check the size
        if roi_size[0] <= 1 or roi_size[1] <= 1:
            continue

        # Edge similarity
        img_direction = v_imgs[img_names[id_view]][1]
        edge_similarity, edge_magnitude = compute_direction_similarity(projected_segment_per_view, img_direction)
        edge_similarities.append(edge_similarity)
        edge_magnitudes.append(edge_magnitude)

        # ROI region
        roi_resized = extract_roi_region(img, roi_center, roi_angle, roi_size, id_view)
        roi_regions.append(roi_resized)

        # LBD descriptor
        descriptor = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
        descriptor = descriptor.compute(img, keyline_from_seg(ori_coords.reshape((1, 4))))[1]
        # descriptor = pytlbd.lbd_single_scale(img, ori_coords.reshape((1, 4)), 9, 7)
        lbd_descriptors.append(descriptor)
    if len(roi_regions) <= 1:
        final_ncc = 0.
        final_edge_similarity = 0.
        final_edge_magnitude = 0.
        lbd_similarity = 0.
    else:
        nccs = calculate_ncc_batch(np.stack(roi_regions, axis=0))  # [-1,1]
        final_ncc = np.mean(nccs) / 2 + 0.5  # [0,1]

        lbd_similarities = calculate_lbd_score_batch(np.stack(lbd_descriptors, axis=0).astype(np.float32) / 255.)
        lbd_similarity = 1 - np.mean(lbd_similarities)

        final_edge_similarity = np.mean(edge_similarities)  # [0,1]
        final_edge_magnitude = np.mean(edge_magnitudes)  # [0,1]
        pass
    return final_ncc, final_edge_similarity, final_edge_magnitude, lbd_similarity


@ray.remote
def compute_loss(
        v_id_worker,
        v_num_sample_per_worker,
        v_segment,
        v_segment_visibility,
        v_projections, v_img_names, v_imgs,
        progress_actor: ray.actor.ActorHandle,
        v_is_dummy: bool
):
    start_index = v_id_worker * v_num_sample_per_worker
    end_index = (v_id_worker + 1) * v_num_sample_per_worker
    if end_index > v_segment.shape[0]:
        end_index = v_segment.shape[0]
    real_num = end_index - start_index
    gt_loss = np.zeros((real_num, 4), dtype=np.float16)
    projected_points = [[] for _ in range(real_num)]
    for id_local in range(v_num_sample_per_worker):
        id_global = v_id_worker * v_num_sample_per_worker + id_local
        if id_global >= v_segment.shape[0]:
            break
        segment = v_segment[id_global].astype(np.float32)
        visibility_mask = v_segment_visibility[:, id_global]

        img_names = [item for idx, item in enumerate(v_img_names) if visibility_mask[idx]]
        projection_matrix_ = [item for idx, item in enumerate(v_projections) if visibility_mask[idx]]

        # Project the segments on image
        homogeneous_sample_segment = np.insert(segment, 3, 1, axis=1)
        num_view = len(img_names)
        # We should at least have two views to reconstruct a target
        # or we will predict it as a negative sample
        if num_view <= 1:
            projected_segment = np.zeros((0, 2, 2))
            final_ncc = 0.
            final_edge_similarity = 0.
            final_edge_magnitude = 0.
            lbd_similarity = 0.
        else:
            # Calculate the projected segment
            projected_segment = np.matmul(np.asarray(projection_matrix_),
                                          np.transpose(homogeneous_sample_segment)).swapaxes(1, 2)
            projected_segment = projected_segment[:, :, :2] / projected_segment[:, :, 2:3]

            # Make sure the start point of the segment is on the left
            is_y_larger_than_x = projected_segment[:, 0, 0] > projected_segment[:, 1, 0]
            projected_segment[is_y_larger_than_x] = projected_segment[is_y_larger_than_x][:, ::-1]

            final_ncc, final_edge_similarity, final_edge_magnitude, lbd_similarity = compute_loss_item(v_imgs,
                                                                                                       num_view,
                                                                                                       projected_segment,
                                                                                                       img_names)
        gt_loss[id_local] = final_ncc, final_edge_similarity, final_edge_magnitude, lbd_similarity
        projected_points[id_local] = projected_segment
        if not v_is_dummy and v_num_sample_per_worker // 100 > 0 and id_local % (v_num_sample_per_worker // 100) == 0:
            progress_actor.report_progress.remote(v_id_worker, id_local)

    if not v_is_dummy:
        os.makedirs("output/gt_loss/sub/", exist_ok=True)
        np.save("output/gt_loss/sub/projected_segments_{}".format(v_id_worker), np.array(projected_points, dtype=object))
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
