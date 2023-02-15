from typing import List

import cv2
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.dataset import Image, Point_3d

from functools import partial
from multiprocessing import Pool
from typing import List

import open3d as o3d
import numpy as np
import os

from tqdm import tqdm

from src.neural_recon.dataset import Image, Point_3d

import pysdf

from src.neural_recon.sample import sample_points_cpu


def check_visibility(v_img: Image, v_points):
    points = np.concatenate([v_points, np.ones_like(v_points[:,0:1])], axis=1)
    pixel_pos = np.matmul(v_img.extrinsic, points.T).T
    is_locate_at_front = pixel_pos[:,2] > 0
    pixel_pos = np.matmul(v_img.intrinsic, pixel_pos.T).T
    pixel_pos = pixel_pos[:,:2] / (pixel_pos[:,2:3] + 1e-8)
    return np.logical_and(is_locate_at_front, np.logical_and(
        pixel_pos > 0,
        pixel_pos < 1,
    ).all(axis = 1))

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
    pos = (pos - bounds_center) / bounds_size * 2
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

    item.line_field_path = os.path.join(v_segments_dir, "line_field_{}.tiff".format(id_img))
    # img = cv2.imread(item.line_field_path, cv2.IMREAD_UNCHANGED)
    # item.line_field = img
    return item


def read_dataset(v_colmap_dir, v_bounds):
    # v_colmap_dir = v_params["dataset"]["colmap_dir"]
    # v_superglue_dir = v_params["dataset"]["superglue_dir"]
    v_segments_dir = os.path.join(v_colmap_dir, "segments")
    bounds_center = (v_bounds[0] + v_bounds[1]) / 2
    bounds_size = (v_bounds[1] - v_bounds[0]).max()

    model_matrix = np.zeros((4, 4), dtype=np.float32)
    model_matrix[0, 0] = bounds_size
    model_matrix[1, 1] = bounds_size
    model_matrix[2, 2] = bounds_size
    model_matrix[0, 3] = bounds_center[0] - bounds_size / 2
    model_matrix[1, 3] = bounds_center[1] - bounds_size / 2
    model_matrix[2, 3] = bounds_center[2] - bounds_size / 2
    model_matrix[3, 3] = 1

    def is_inside_scene(v_pos):
        return v_pos[0] > -1 and v_pos[1] > -1 and v_pos[2] > -1 and v_pos[0] < 1 and v_pos[1] < 1 and v_pos[2] < 1

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
            camera_intrinsic[cam_id]["img_size"] = (width,height)
    print("Found {} cameras".format(len(camera_intrinsic)))

    imgs: List[Image] = []
    original_img_id_to_current_img_id = {}
    with open(os.path.join(v_colmap_dir, "images.txt")) as f:
        data = [item.strip().split(" ") for item in f.readlines() if item[0] != "#"]
        assert len(data) % 2 == 0
        imgs = [Image(-1, None, None, None, None) for _ in range(len(data) // 2)]
        for id_img in tqdm(range(len(imgs))):
            line = data[id_img * 2]
            imgs[id_img].id_img, qw, qx, qy, qz, tx, ty, tz, camID, img_name = \
                int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(
                    line[6]), float(line[7]), int(line[8]), line[9]
            original_img_id_to_current_img_id[imgs[id_img].id_img] = id_img
            imgs[id_img].img_path = os.path.join(v_colmap_dir, "../imgs/", img_name)
            extrinsic = quaternion_rotation_matrix([qw, qx, qy, qz])
            extrinsic = np.concatenate([extrinsic, np.array([[tx, ty, tz]]).transpose()], axis=1, dtype=np.float32)
            extrinsic_homo = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
            projection_matrix = np.zeros((4, 4), dtype=np.float32)
            projection_matrix[:3, :3] = camera_intrinsic[cam_id]["normalized_K"]
            projection_matrix[3, 3] = 1
            # Rescale
            extrinsic_homo = np.matmul(extrinsic_homo, model_matrix)
            projection_matrix = np.matmul(projection_matrix, extrinsic_homo)
            imgs[id_img].intrinsic = camera_intrinsic[cam_id]["normalized_K"]
            imgs[id_img].extrinsic = extrinsic
            imgs[id_img].projection = projection_matrix
            imgs[id_img].img_size = camera_intrinsic[camID]["img_size"]

    print("Found {} viewpoints".format(len(imgs)))

    points_3d: List[Point_3d] = []
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
    imgs = list(pool.map(partial(read_segments, v_segments_dir), enumerate(imgs)))
    pool.close()

    # Testcase
    if False:
        id_point = 0
        point_3d = points_3d[id_point]
        pos_3d = point_3d.pos
        id_img = point_3d.tracks[0][0]
        id_keypoint = point_3d.tracks[0][1]
        img = np.asarray(cv2.imread(imgs[id_img].img_path)).copy()
        projection = imgs[id_img].projection
        pos_2d = np.matmul(projection, np.concatenate([pos_3d, np.ones_like(pos_3d[0:1])], axis=0))
        pos_2d = pos_2d[:2] / pos_2d[2] / pos_2d[3]
        pos_2d[0] *= imgs[id_img].img_size[0]
        pos_2d[1] *= imgs[id_img].img_size[1]
        img = cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), 10, (0, 0, 255), 10)
        # for item in imgs[id_img].detected_points:
        #     img = cv2.circle(img, (int(item[0] * imgs[id_img].img_size[0]), int(item[1] * imgs[id_img].img_size[1])), 10, (0, 255, 255), 10)
        cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1", 1600, 900)
        cv2.imshow("1", img)
        cv2.waitKey()

    return imgs, points_3d


class Colmap_dataset(torch.utils.data.Dataset):
    def __init__(self, v_colmap_dir_path, v_bounds, v_mode, v_mesh):
        super(Colmap_dataset, self).__init__()
        self.trainer_mode = v_mode

        # Read colmap and segments
        imgs, world_points = read_dataset(v_colmap_dir_path, v_bounds)

        self.imgs: List[Image] = imgs
        self.world_points: List[Point_3d] = world_points

        # Read mesh and normalize it
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(v_colmap_dir_path,"gt_mesh.ply"))
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)

        self.bounds_center = (v_bounds[0] + v_bounds[1]) / 2
        self.bounds_size = (v_bounds[1] - v_bounds[0]).max()
        vertices = (vertices - self.bounds_center) / self.bounds_size + 0.5

        if True:
            sdf_computer=pysdf.PYSDF_computer()
            sdf_computer.setup_bounds(
                np.append(self.bounds_center, self.bounds_size)
            )
            sdf_computer.setup_mesh(vertices[faces], False)
            sdf = sdf_computer.compute_sdf(int(1e0), int(1e6), int(1e6), False)

            sample_points = sdf[:, :3]
            sample_distances = sdf[:, 3:]

            pool = Pool(16)
            visibility_window = pool.map(
                partial(check_visibility, v_points=(sample_points-0.5) * self.bounds_size + self.bounds_center),
                imgs, chunksize=10)
            visibility_window = np.stack(visibility_window, axis=0)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(sample_points)
            o3d.io.write_point_cloud("output/1.ply", pc)

            pc.points = o3d.utility.Vector3dVector(sample_points[visibility_window[0]])
            o3d.io.write_point_cloud("output/2.ply", pc)

            visibility_intersection = sdf_computer.check_visibility(
                (np.asarray([item.extrinsic[:3,3] for item in imgs]) - sdf_computer.m_center) / sdf_computer.m_scale + 0.5,
                sdf[:,:3]
            )
            visibility_intersection=visibility_intersection.reshape([len(imgs), -1]).astype(np.bool)
            pc.points = o3d.utility.Vector3dVector(sample_points[visibility_intersection[0]==1])
            o3d.io.write_point_cloud("output/3.ply", pc)

            self.final_visibility = np.logical_and(visibility_window, visibility_intersection)
            pc.points = o3d.utility.Vector3dVector(sample_points[self.final_visibility[0]])
            o3d.io.write_point_cloud("output/4.ply", pc)
            pass

        pass

    def __getitem__(self, index):
        index = 1
        point_3d = self.world_points[index]
        projection_matrix_ = np.zeros((len(point_3d.tracks), 4, 4), dtype=np.float32)
        id_imgs: torch.Tensor = torch.zeros(len(point_3d.tracks), dtype=torch.long)
        keypoints_: List[torch.Tensor] = []
        for id_track, track in enumerate(point_3d.tracks):
            id_img = track[0]
            id_imgs[id_track] = id_img
            projection_matrix_[id_track] = self.imgs[id_img].projection
            keypoints_.append(torch.from_numpy(self.imgs[id_img].detected_points))

        projection_matrix = torch.from_numpy(projection_matrix_)
        keypoints = pad_sequence(keypoints_, batch_first=True, padding_value=-1)
        keypoints = torch.cat([keypoints, torch.logical_not(torch.all(keypoints == -1, dim=2, keepdim=True))], dim=2)

        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["id_imgs"] = id_imgs
        data["keypoints"] = keypoints
        data["projection_matrix"] = projection_matrix
        return data

    def __len__(self):
        return len(self.world_points)

    @staticmethod
    def collate_fn(batch):
        id_points = [item["id"] for item in batch]
        id_imgs_ = [item["id_imgs"] for item in batch]
        keypoints_ = [item["keypoints"] for item in batch]
        projection_matrix_ = [item["projection_matrix"] for item in batch]

        # keypoints = pad_sequence(keypoints_,batch_first=True,padding_value=-1)
        id_imgs = pad_sequence(id_imgs_, batch_first=True, padding_value=-1)
        projection_matrix = pad_sequence(projection_matrix_, batch_first=True, padding_value=-1)
        valid_views = torch.logical_not(torch.all(torch.flatten(projection_matrix, start_dim=2) == -1, dim=2))

        return {
            'id_points': torch.stack(id_points, dim=0),
            'id_imgs': id_imgs,
            'keypoints': keypoints_,
            'projection_matrix': projection_matrix,
            'valid_views': valid_views,
        }
