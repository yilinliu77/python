from typing import Tuple

import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import cv2
import numba as nb

# The similarity is between 0 and 1
# Larger is better
def compute_direction_similarity(v_segment: np.ndarray, v_direction_img: np.ndarray):
    source_direction = v_segment[1] - v_segment[0]
    delta = source_direction / 10
    sample_points = v_segment[0] + np.transpose(delta[:, np.newaxis] * np.arange(11))
    sample_points_index = (sample_points * v_direction_img.shape[:2][::-1]).astype(int)
    sample_direction = v_direction_img[sample_points_index[:, 1], sample_points_index[:, 0]]
    angle_cos1 = np.abs(np.dot(sample_direction, source_direction) / (
            1e-8 + np.linalg.norm(sample_direction, axis=1) * np.linalg.norm(source_direction)))
    angle_cos2 = np.abs(np.dot(sample_direction, -source_direction) / (
            1e-8 + np.linalg.norm(sample_direction, axis=1) * np.linalg.norm(-source_direction)))
    angle_cos = np.min(np.asarray((angle_cos1, angle_cos2)), axis=0)
    return 1 - np.mean(angle_cos), np.mean(np.linalg.norm(sample_direction,axis=1)) / 380. / 2. + 0.5


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def calculate_ncc_batch(v_roi_regions):
    nccs = []
    for id_view1 in range(len(v_roi_regions)):
        for id_view2 in range(id_view1 + 1, len(v_roi_regions)):
            ncc = correlation_coefficient(v_roi_regions[id_view1], v_roi_regions[id_view2])
            nccs.append(ncc)
    return nccs


# Compute the related region using a rotated rectangle
def extract_roi_rectangle(v_normalized_segment: np.ndarray, v_img_size: Tuple):
    img_size_ = np.asarray(v_img_size, np.float32)
    projected_segment_per_view_original = v_normalized_segment * img_size_
    roi_size = (
        int(np.linalg.norm((projected_segment_per_view_original[1, :] - projected_segment_per_view_original[0, :]),
                           None,0).item()), 10)
    roi_angle = np.arctan2(
        projected_segment_per_view_original[1, 1] - projected_segment_per_view_original[0, 1],
        projected_segment_per_view_original[1, 0] - projected_segment_per_view_original[0, 0])
    roi_center = (projected_segment_per_view_original.sum(axis=0)) / 2

    return roi_size, roi_angle, roi_center


def extract_roi_region(v_img: np.ndarray, v_roi_center, v_roi_angle, v_roi_size, id_view):
    M = cv2.getRotationMatrix2D(v_roi_center, np.rad2deg(v_roi_angle), 1)
    img_rotated = cv2.warpAffine(v_img, M, v_img.shape[:2][::-1], cv2.INTER_CUBIC)
    roi = cv2.getRectSubPix(img_rotated, v_roi_size, v_roi_center)
    roi_resized = cv2.resize(roi, (50, 10), interpolation=cv2.INTER_AREA)
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
