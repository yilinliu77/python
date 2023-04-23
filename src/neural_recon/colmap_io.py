from dataclasses import dataclass

import cv2
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import open3d as o3d
import numpy as np
import numba as nb
import os

from tqdm import tqdm


@dataclass
class Image:
    id_img: int
    img_name: str
    img_path: str
    pos: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    projection: np.ndarray
    detected_points: np.ndarray = np.zeros((1, 1))
    detected_lines: np.ndarray = np.zeros((1, 1))
    line_field: np.ndarray = np.zeros((1, 1))
    line_field_path: str = ""
    img_size: tuple[int, int] = (-1, -1)


@dataclass
class Point_3d:
    pos: np.ndarray
    tracks: (int, int)

@nb.njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


@nb.njit
def check_visibility(v_projection: np.ndarray, v_points: np.ndarray):
    _ones = np.ones_like(v_points[:, 0:1])
    points = np.concatenate((v_points, _ones), axis=1)
    pixel_pos = np.dot(v_projection, np.ascontiguousarray(np.transpose(points)))
    pixel_pos = np.transpose(pixel_pos)
    is_locate_at_front = pixel_pos[:, 2] > 0
    # pixel_pos = np.matmul(v_img.intrinsic, pixel_pos.T).T
    pixel_pos = pixel_pos[:, :2] / pixel_pos[:, 2:3] / pixel_pos[:, 3:4]

    is_within_window_ = np.logical_and(
        pixel_pos > 0,
        pixel_pos < 1,
    )
    is_within_window = np_all_axis1(is_within_window_)
    return np.logical_and(is_locate_at_front, is_within_window)


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def read_world_points(original_img_id_to_current_img_id, bounds_center, bounds_size, line):
    pos = np.array([float(line[1]), float(line[2]), float(line[3])], dtype=np.float32)
    pos = (pos - bounds_center) / bounds_size + 0.5
    assert (len(line) - 8) % 2 == 0
    num_track = (len(line) - 8) // 2
    tracks = []
    for i_track in range(num_track):
        tracks.append(
            (original_img_id_to_current_img_id[int(line[8 + 2 * i_track + 0])], int(line[8 + 2 * i_track + 1])))
    return Point_3d(pos, tracks)


def read_superpoints(v_superglue_dir, v_img_size, v_args):
    id_img, img_item = v_args
    with open(os.path.join(v_superglue_dir, os.path.basename(img_item.img_path) + ".txt")) as f:
        line = [item.strip().split(" ") for item in f.readlines()]
    line.pop(0)
    num_point = len(line)
    img_item.detected_points = np.zeros((num_point, 2), np.float32)
    for i_segment in range(num_point):
        img_item.detected_points[i_segment, 0] = float(line[i_segment][0]) / v_img_size[0]
        img_item.detected_points[i_segment, 1] = float(line[i_segment][1]) / v_img_size[1]
    return img_item


def read_segments(v_segments_dir, v_args):
    id_img, item = v_args
    v_img_size = item.img_size
    with open(os.path.join(v_segments_dir, "detected_lines_{}.txt".format(id_img))) as f:
        line = f.readline().strip().split(" ")[5:]
    num_segments = int(line[0])
    item.detected_lines = np.zeros((num_segments, 4), np.float32)
    for i_segment in range(num_segments):
        item.detected_lines[i_segment, 0] = float(line[4 + i_segment * 6 + 2]) / v_img_size[0]
        item.detected_lines[i_segment, 1] = float(line[4 + i_segment * 6 + 3]) / v_img_size[1]
        item.detected_lines[i_segment, 2] = float(line[4 + i_segment * 6 + 4]) / v_img_size[0]
        item.detected_lines[i_segment, 3] = float(line[4 + i_segment * 6 + 5]) / v_img_size[1]

    item.line_field_path = os.path.join(v_segments_dir, "{}.tiff".format(item.img_name))
    if os.path.exists(item.line_field_path):
        img = cv2.imread(item.line_field_path, cv2.IMREAD_UNCHANGED)
        item.line_field = img
    return item


def read_dataset(v_colmap_dir, v_bounds):
    # v_colmap_dir = v_params["dataset"]["colmap_dir"]
    # v_superglue_dir = v_params["dataset"]["superglue_dir"]
    v_segments_dir = os.path.join(v_colmap_dir, "segments")
    bounds_center = (v_bounds[0] + v_bounds[1]) / 2
    bounds_size = (v_bounds[1] - v_bounds[0]).max()

    model_matrix = np.zeros((4, 4),dtype=np.float32)
    model_matrix[0, 0] = bounds_size
    model_matrix[1, 1] = bounds_size
    model_matrix[2, 2] = bounds_size
    model_matrix[0, 3] = bounds_center[0] - bounds_size / 2
    model_matrix[1, 3] = bounds_center[1] - bounds_size / 2
    model_matrix[2, 3] = bounds_center[2] - bounds_size / 2
    model_matrix[3, 3] = 1

    def is_inside_scene(v_pos):
        return v_pos[0] > 0 and v_pos[1] > 0 and v_pos[2] > 0 and v_pos[0] < 1 and v_pos[1] < 1 and v_pos[2] < 1

    pool = Pool(16)

    # Read 3D points in world coordinate as the optimization target
    camera_intrinsic: dict = {}
    with open(os.path.join(v_colmap_dir, "cameras.txt")) as f:
        data = [item.strip().split(" ") for item in f.readlines()]
        for line in data:
            if line[0] == "#":
                continue
            cam_id, camera_model, width, height = int(line[0]), line[1], int(line[2]), int(line[3])
            if camera_model != "SIMPLE_PINHOLE":
                raise "Unsupported camera model"
            fx, cx, cy = float(line[4]), float(line[5]), float(line[6])
            fy = fx
            k1 = 0
            k2 = 0
            k3 = 0
            p1 = 0
            p2 = 0
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = fx / width
            K[0, 1] = 0
            K[0, 2] = cx / width
            K[1, 0] = 0
            K[1, 1] = fy / height
            K[1, 2] = cy / height
            K[2, 0] = 0
            K[2, 1] = 0
            K[2, 2] = 1
            camera_intrinsic[cam_id] = {}
            camera_intrinsic[cam_id]["normalized_K"] = K
            camera_intrinsic[cam_id]["img_size"] = (width, height)
    print("Found {} cameras".format(len(camera_intrinsic)))

    imgs: List[Image] = []
    original_img_id_to_current_img_id = {}
    with open(os.path.join(v_colmap_dir, "images.txt")) as f:
        data = [item.strip().split(" ") for item in f.readlines() if item[0] != "#"]
        assert len(data) % 2 == 0
        imgs = [Image(-1, None, None, None, None, None, None) for _ in range(len(data) // 2)]
        for id_img in tqdm(range(len(imgs))):
            line = data[id_img * 2]
            imgs[id_img].id_img, qw, qx, qy, qz, tx, ty, tz, camID, img_name = \
                int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(
                    line[6]), float(line[7]), int(line[8]), line[9]
            original_img_id_to_current_img_id[imgs[id_img].id_img] = id_img
            imgs[id_img].img_path = os.path.join(v_colmap_dir, "imgs/", img_name)
            extrinsic = quaternion_rotation_matrix([qw, qx, qy, qz])
            extrinsic = np.concatenate([extrinsic, np.array([[tx, ty, tz]]).transpose()], axis=1, dtype=np.float32)
            extrinsic_homo = np.concatenate([extrinsic, np.array([[0., 0., 0., 1.]])], axis=0, dtype=np.float32)
            extrinsic_homo_inv = np.linalg.inv(extrinsic_homo)
            imgs[id_img].pos = extrinsic_homo_inv[:3, 3]
            imgs[id_img].pos = (imgs[id_img].pos - bounds_center) / bounds_size + 0.5
            projection_matrix = np.zeros((4, 4), dtype=np.float32)
            projection_matrix[:3, :3] = camera_intrinsic[cam_id]["normalized_K"]
            projection_matrix[3, 3] = 1
            # Rescale
            extrinsic_homo = np.matmul(extrinsic_homo, model_matrix)
            projection_matrix = np.matmul(projection_matrix, extrinsic_homo)
            imgs[id_img].intrinsic = camera_intrinsic[cam_id]["normalized_K"]
            imgs[id_img].extrinsic = extrinsic_homo
            imgs[id_img].projection = projection_matrix
            imgs[id_img].img_size = camera_intrinsic[camID]["img_size"]
            imgs[id_img].img_name=os.path.basename(imgs[id_img].img_path).split(".")[0]

    print("Found {} viewpoints".format(len(imgs)))

    points_3d: List[Point_3d] = []
    if True: # For now we don't need it
        with open(os.path.join(v_colmap_dir, "points3D.txt")) as f:
            data = [item.strip().split(" ") for item in f.readlines() if item[0] != "#"]
            points_3d = list(pool.map(
                partial(read_world_points, original_img_id_to_current_img_id, bounds_center, bounds_size),
                data, chunksize=1000))
        num_original_points = len(points_3d)
        points_3d = list(filter(lambda item: is_inside_scene(item.pos), points_3d))

        print(
            "Found {} world points after filtering {} points".format(len(points_3d), num_original_points - len(points_3d)))

    # Read point field and line field for each img
    # imgs = list(pool.map(partial(read_superpoints, v_superglue_dir, img_size), enumerate(imgs)))
    # imgs = list(pool.map(partial(read_segments, v_segments_dir), enumerate(imgs)))
    # print("Done reading segments")
    pool.close()

    # Testcase
    if False:
        cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1", 1600, 900)
        cv2.moveWindow("1", 0, 0)

        id_points = [10]
        for id_point in id_points:
            point_3d = points_3d[id_point]
            pos_3d = point_3d.pos
            original_pos3d=(pos_3d - 0.5) * bounds_size + bounds_center
            print("\nPoint: {} has {} cameras".format(original_pos3d, len(point_3d.tracks)))
            # pos_3d = (pos_3d - 0.5) * bounds_size + bounds_center
            for track in point_3d.tracks:
                id_img = track[0]
                id_keypoint = track[1]
                img = np.asarray(cv2.imread(imgs[id_img].img_path)).copy()
                projection = imgs[id_img].projection
                pos_2d = np.matmul(projection, np.concatenate([pos_3d, np.ones_like(pos_3d[0:1])], axis=0))
                pos_2d = pos_2d[:2] / pos_2d[2] / pos_2d[3]
                pos_2d[0] *= imgs[id_img].img_size[0]
                pos_2d[1] *= imgs[id_img].img_size[1]
                img = cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), 10, (0, 0, 255), 10)
                # for item in imgs[id_img].detected_points:
                #     img = cv2.circle(img, (int(item[0] * imgs[id_img].img_size[0]), int(item[1] * imgs[id_img].img_size[1])), 10, (0, 255, 255), 10)
                print("---- {}".format((imgs[id_img].pos - 0.5) * bounds_size + bounds_center))
                cv2.imshow("1", img)
                cv2.waitKey()

    return imgs, points_3d

