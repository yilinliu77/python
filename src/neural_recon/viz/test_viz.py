import math
import random
import sys, os

import open3d

from shared.perspective_geometry import extract_fundamental_from_projection
from src.neural_recon.colmap_io import read_dataset, Image, Point_3d

sys.path.append("thirdparty/sdf_computer/build/")

import cv2
import numpy as np
import torch
from tqdm import tqdm
from shared.common_utils import to_homogeneous, save_line_cloud, normalize_vector, normalized_torch_img_to_numpy
from src.neural_recon.optimize_segment import optimize_single_segment, optimize_single_segment_tensor
from src.neural_recon.bak.phase1 import Phase1
import faiss

if __name__ == '__main__1':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    v_data_dir = "output/segment_104"
    raw_data_npy = os.path.join(v_data_dir, "data.npy")
    gt_loss_npy = os.path.join(v_data_dir, "gt_loss.npy")
    imgs_npy = os.path.join(v_data_dir, "imgs.npy")
    projected_points_npy = os.path.join(v_data_dir, "projected_segments.npy")
    raw_data = np.load(raw_data_npy, allow_pickle=True)[()]
    gt_loss = np.load(gt_loss_npy)
    original_imgs = np.load(imgs_npy, allow_pickle=True)[()]
    points_from_sfm_2d = np.load(projected_points_npy, allow_pickle=True)[()]

    id_viz = 11
    for id_viz in tqdm(range(points_from_sfm_2d.shape[0])):
        sample_point = raw_data["segments"][id_viz]
        projected_coordinates = (points_from_sfm_2d[id_viz] * np.array((600, 400))).astype(np.int64)
        imgs = [original_imgs[item] for idx, item in enumerate(original_imgs) if
                raw_data["segments_visibility"][idx, id_viz]]
        viz_imgs = []
        if len(imgs) <= 1: continue
        height = imgs[0][0].shape[0]
        width = imgs[0][0].shape[1]
        new_img = np.zeros((height * 2, width * 2), np.uint8)
        for id_img, item in enumerate(imgs):
            img = item[0].copy()
            viz_img = cv2.line(img,
                               projected_coordinates[id_img][0],
                               projected_coordinates[id_img][1],
                               (0, 0, 255), 2)
            if id_img == 0:
                new_img[:height, :width] = viz_img
            elif id_img == 1:
                new_img[:height, width:] = viz_img
            elif id_img == 2:
                new_img[height:, :width] = viz_img
            elif id_img == 3:
                new_img[height:, width:] = viz_img
        content = "{}:{:.4f}\n{}:{:.4f}\n{}:{:.4f}\n{}:{:.4f}\n".format(
            "NCC", gt_loss[id_viz, 0].item(), "Edge Similarity", gt_loss[id_viz, 1].item(), "Edge Magnitude",
            gt_loss[id_viz, 2].item(), "LBD", gt_loss[id_viz, 3].item()
        )

        y0, dy = 50, 4
        fontFace = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 0.5
        thickness = 1
        for i, line in enumerate(content.split('\n')):
            y = y0 + i * dy
            cv2.putText(new_img, line, (10, y), fontFace, fontScale, (0, 0, 255), thickness)
            dy = cv2.getTextSize(line, fontFace, fontScale, thickness)[0][1] + 10
        cv2.imwrite("output/viz_segment/{:08}.png".format(id_viz), new_img)

    cv2.imshow("1", viz_imgs)
    cv2.waitKey()
    pass

if __name__ == '__main__3':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    v_data_dir = "output/points_106"
    raw_data_npy = os.path.join(v_data_dir, "data.npy")
    gt_loss_npy = os.path.join(v_data_dir, "gt_loss.npy")
    imgs_npy = os.path.join(v_data_dir, "imgs.npy")
    projected_points_npy = os.path.join(v_data_dir, "projected_points.npy")
    raw_data = np.load(raw_data_npy, allow_pickle=True)[()]
    gt_loss = np.load(gt_loss_npy)
    original_imgs = np.load(imgs_npy, allow_pickle=True)[()]
    points_from_sfm_2d = np.load(projected_points_npy, allow_pickle=True)[()]

    id_viz = 11
    candidates = np.random.randint(0, points_from_sfm_2d.shape[0], 10000)
    for id_viz in tqdm(candidates):
        sample_point = raw_data["sample_points"][id_viz]
        projected_coordinates = (points_from_sfm_2d[id_viz] * np.array((600, 400))).astype(np.int64)
        imgs = [original_imgs[item] for idx, item in enumerate(original_imgs) if
                raw_data["final_visibility"][idx, id_viz]]
        viz_imgs = []
        if len(imgs) <= 1: continue
        height = imgs[0].shape[0]
        width = imgs[0].shape[1]
        new_img = np.zeros((height * 2, width * 2, 3), np.uint8)
        for id_img, item in enumerate(imgs):
            img = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
            viz_img = cv2.circle(img,
                                 projected_coordinates[id_img], 3,
                                 (0, 0, 255), 2)
            if id_img == 0:
                new_img[:height, :width] = viz_img
            elif id_img == 1:
                new_img[:height, width:] = viz_img
            elif id_img == 2:
                new_img[height:, :width] = viz_img
            elif id_img == 3:
                new_img[height:, width:] = viz_img
        content = "{}:{:.4f}".format(
            "NCC", gt_loss[id_viz, 0].item()
        )

        y0, dy = 50, 4
        fontFace = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 1
        thickness = 1
        cv2.putText(new_img, content, (10, 50), fontFace, fontScale, (0, 0, 255), thickness)
        cv2.imwrite("output/viz_point/{:08}.png".format(id_viz), new_img)

    cv2.imshow("1", viz_imgs)
    cv2.waitKey()
    pass

if __name__ == '__main__33':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    v_data_dir = "output/points_104"
    raw_data_npy = os.path.join(v_data_dir, "data.npy")
    gt_loss_npy = os.path.join(v_data_dir, "gt_loss.npy")
    imgs_npy = os.path.join(v_data_dir, "imgs.npy")
    projected_points_npy = os.path.join(v_data_dir, "projected_points.npy")
    raw_data = np.load(raw_data_npy, allow_pickle=True)[()]
    gt_loss = np.load(gt_loss_npy)
    original_imgs = np.load(imgs_npy, allow_pickle=True)[()]
    points_from_sfm_2d = np.load(projected_points_npy, allow_pickle=True)[()]

    id_viz = 1
    candidates = np.random.randint(0, 10000, 10000)
    for id_viz in candidates:
        sample_point = raw_data["sample_points"][id_viz]
        projected_coordinates = points_from_sfm_2d[id_viz] * np.array((600, 400))
        img_database = [raw_data["img_database"][idx] for idx, item in enumerate(original_imgs) if
                        raw_data["final_visibility"][idx, id_viz]]
        imgs = [original_imgs[item] for idx, item in enumerate(original_imgs) if
                raw_data["final_visibility"][idx, id_viz]]
        viz_imgs = []
        if len(imgs) <= 1: continue

        id_img1 = 0
        for id_img2 in range(1, len(imgs)):
            height = imgs[id_img1].shape[0]
            width = imgs[id_img1].shape[1]
            img1 = img_database[id_img1]
            rgb1 = cv2.cvtColor(imgs[id_img1], cv2.COLOR_GRAY2BGR)
            img2 = img_database[id_img2]
            rgb2 = cv2.cvtColor(imgs[id_img2], cv2.COLOR_GRAY2BGR)

            coor1 = np.insert(points_from_sfm_2d[id_viz][id_img1], 2, 1)
            coor2 = np.insert(points_from_sfm_2d[id_viz][id_img2], 2, 1)
            transform12 = img2.extrinsic @ np.linalg.inv(img1.extrinsic)

            point2camera = (img1.extrinsic @ to_homogeneous(sample_point))[:3]
            point2camera_length = np.linalg.norm(point2camera)
            normal = -point2camera / point2camera_length

            H = img2.intrinsic @ (transform12[:3, :3] - 1 / point2camera_length * transform12[:3, 3:4] @ np.transpose(
                normal[:, np.newaxis])) @ np.linalg.inv(img1.intrinsic)

            coor1_int = (coor1[:2] * np.array((600, 400))).astype(np.int32)
            coor2_int = (coor2[:2] * np.array((600, 400))).astype(np.int32)

            half_window_size = 7
            y_min = coor1_int[1] - half_window_size
            y_max = coor1_int[1] + half_window_size
            x_min = coor1_int[0] - half_window_size
            x_max = coor1_int[0] + half_window_size
            vertices1 = np.array((
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ), np.int32)
            cv2.polylines(rgb1, [vertices1], True, (0, 0, 255), 1)
            vertices2 = np.transpose(
                H @ np.transpose(np.insert(vertices1.astype(np.float32) / np.array((600, 400)), 2, 1, axis=1)))
            vertices2 = vertices2[:, :2] / vertices2[:, 2:3]
            vertices2 = (vertices2 * np.array((600, 400))).astype(np.int32)
            cv2.polylines(rgb2, [vertices2], True, (0, 0, 255), 1)

            cv2.imshow("1", np.concatenate((rgb1, rgb2), axis=1))
            cv2.waitKey()
            pass

if __name__ == '__main__':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    bound_min = np.array((-40, -40, -5))
    bound_max = np.array((130, 150, 60))
    bounds_center = (bound_min + bound_max) / 2
    bounds_size = (bound_max - bound_min).max()


    def normalize_segment(v_point):
        return (v_point - bounds_center) / bound_max + 0.5


    if False:
        mesh = open3d.io.read_triangle_mesh(r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Test_imgs2_colmap_neural\sparse_align\detailed_l7_with_ground.ply")
        mesh.vertices = open3d.utility.Vector3dVector((np.asarray(mesh.vertices)-bounds_center)/bounds_size+0.5)
        open3d.io.write_triangle_mesh("output/img_field_test/normalized_mesh_world_coordinate.ply", mesh)

        imgs, points_3d = read_dataset(
            r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Test_imgs2_colmap_neural\sparse_align",
            [bound_min,
             bound_max]
        )
        print("Start read wireframe")
        with open(
                r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Test_imgs2_colmap_neural\sparse_align\segments\wireframe_1.obj") as f:
            data = [line.strip() for line in f.readlines()]
            vertices1 = np.asarray([item[2:-2].split(" ") for item in data if item[0] == "v"], dtype=np.float32)
            lines1 = np.asarray([item[2:].split(" ") for item in data if item[0] == "l"], dtype=np.int32)
        with open(
                r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Test_imgs2_colmap_neural\sparse_align\segments\wireframe_2.obj") as f:
            data = [line.strip() for line in f.readlines()]
            vertices2 = np.asarray([item[2:-2].split(" ") for item in data if item[0] == "v"], dtype=np.float32)
            lines2 = np.asarray([item[2:].split(" ") for item in data if item[0] == "l"], dtype=np.int32)

        print("Start to filter points")
        preserved_points = []
        for point in tqdm(points_3d):
            for track in point.tracks:
                if track[0] in [0, 1, 2]:
                    preserved_points.append(point)

        np.save("output/img_field_test/imgs", imgs[:3], allow_pickle=True)
        np.save("output/img_field_test/points", preserved_points, allow_pickle=True)
        np.save("output/img_field_test/lines", [vertices1, lines1, vertices2, lines2], allow_pickle=True)
    else:
        imgs: list[Image] = np.load("output/img_field_test/imgs.npy", allow_pickle=True).tolist()
        points_3d: list[Point_3d] = np.load("output/img_field_test/points.npy", allow_pickle=True).tolist()
        vertices1, lines1, vertices2, lines2 = np.load("output/img_field_test/lines.npy", allow_pickle=True)
        vertices1 = vertices1 / np.array((1499, 999), )
        vertices2 = vertices2 / np.array((1499, 999), )
        lines1 = lines1 - 1
        lines2 = lines2 - 1

    points_from_sfm = np.stack([item.pos for item in points_3d])

    img1 = imgs[1]
    img2 = imgs[2]
    print("Project all the points on img1")
    def project_points(points_3d_pos):
        projected_points = np.transpose(img1.projection @ np.transpose(np.insert(points_3d_pos, 3, 1, axis=1)))
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        projected_points_mask = np.logical_and(projected_points[:, 0] > 0, projected_points[:, 1] > 0)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 0] < 1)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 1] < 1)
        points_3d_pos = points_3d_pos[projected_points_mask]
        projected_points = projected_points[projected_points_mask]
        return points_3d_pos, projected_points
    points_from_sfm, points_from_sfm_2d = project_points(points_from_sfm)

    rgb1 = cv2.imread(img1.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    rgb2 = cv2.imread(img2.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    shape = rgb1.shape[:2][::-1]

    id_segment = 712
    distance_threshold = 20 / 6000

    lines1 = np.concatenate((vertices1[lines1[:, 0]], vertices1[lines1[:, 1]]), axis=1)
    lines2 = np.concatenate((vertices2[lines2[:, 0]], vertices2[lines2[:, 1]]), axis=1)

    print("Draw keypoints on img1")
    def draw_initial():
        point_img = rgb1.copy()
        for point in points_from_sfm_2d:
            cv2.circle(point_img, (point * shape).astype(np.int32), 5, (0, 0, 255), thickness=10)
        print("Draw lines on img1")
        line_img1 = rgb1.copy()
        line_img2 = rgb2.copy()
        for idx, line in enumerate(lines1):
            # print(idx)
            cv2.line(line_img1, (line[:2] * shape).astype(np.int32), (line[2:] * shape).astype(np.int32),
                     (0, 0, 255),
                     thickness=1, lineType=cv2.LINE_AA)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()
        cv2.line(line_img1, (lines1[id_segment, :2] * shape).astype(np.int32),
                 (lines1[id_segment, 2:] * shape).astype(np.int32), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        for idx, line in enumerate(lines2):
            cv2.line(line_img2, (line[:2] * shape).astype(np.int32), (line[2:] * shape).astype(np.int32),
                     (0, 0, 255),
                     thickness=1, lineType=cv2.LINE_AA)

        viz_img = np.concatenate((point_img, line_img1, line_img2), axis=0)
        cv2.imwrite("output/img_field_test/input_img1.jpg", viz_img)
    draw_initial()

    print("Build kd tree and compute initial line clouds")
    def compute_initial(id_segment):
        distance_threshold = 5 # 10m

        # Query segments: (M, 4)
        # points from sfm: (N, 2)
        kd_tree = faiss.IndexFlatL2(2)
        kd_tree.add(points_from_sfm_2d.astype(np.float32))
        query_points = lines1.reshape(-1, 2)  # (2*M, 2)
        shortest_distance, index_shortest_distance = kd_tree.search(query_points, 8) # (2*M, K)

        points_from_sfm_camera = np.transpose(img1.extrinsic @ np.transpose(np.insert(points_from_sfm,3,1,axis=1)))[:,:3] # (N, 3)

        # Select the point which is nearest to the actual ray for each endpoints
        # 1. Construct the ray
        pc = np.transpose(np.linalg.inv(img1.intrinsic) @ np.transpose(np.insert(query_points, 2, 1, axis=1))) # (2M, 2); points in camera coordinates
        pc = pc / np.linalg.norm(pc,axis=1, keepdims=True) # Normalize the points
        nearest_candidates = points_from_sfm_camera[index_shortest_distance] # (2M, K, 3)
         # Compute the shortest distance from the candidate point to the ray for each query point
        distance_of_projection = nearest_candidates @ pc[:,:,np.newaxis] # (2M, K, 1): K projected distance of the candidate point along each ray
        projected_points_on_ray = distance_of_projection * pc[:,np.newaxis,:] # (2M, K, 3): K projected points along the ray
        distance_from_candidate_points_to_ray = np.linalg.norm(nearest_candidates - projected_points_on_ray, axis=2) # (2M, 1)
        index_best_projected = distance_from_candidate_points_to_ray.argmin(axis=1) # (2M, 1): Index of the best projected points along the ray

        chosen_distances = distance_of_projection[np.arange(projected_points_on_ray.shape[0]),index_best_projected]
        valid_mask = distance_from_candidate_points_to_ray[np.arange(projected_points_on_ray.shape[0]),index_best_projected] < distance_threshold # (2M, 1)
        valid_mask = valid_mask.reshape((-1,2)) # (M, 2)
        valid_mask = np.logical_and(valid_mask[:,0],valid_mask[:,1]) # (M,)
        initial_points_camera = projected_points_on_ray[np.arange(projected_points_on_ray.shape[0]),index_best_projected] # (2M, 3): The best projected points along the ray
        initial_points_world=np.transpose(np.linalg.inv(img1.extrinsic) @ np.transpose(np.insert(initial_points_camera, 3, 1, axis=1)))
        initial_points_world=initial_points_world[:,:3]/initial_points_world[:,3:4]

        line_coordinates = initial_points_world.reshape(-1, 6) # (M, 6)
        line_coordinates = line_coordinates[valid_mask]
        valid_distances = chosen_distances.reshape(-1, 2)
        valid_distances = valid_distances[valid_mask]

        # line_coordinates = points_from_sfm[index_shortest_distance[:, 0], :].reshape(-1, 6)
        id_segment = (valid_mask[:id_segment]).sum()

        save_line_cloud("output/img_field_test/initial_segments.obj", line_coordinates)
        return id_segment, line_coordinates, valid_distances
    changed_id_segment, segment_3d_coordinates, segment_3d_distances = compute_initial(id_segment) # Coordinate of the initial segments in world coordinate


    img_model_root_dir = r"D:\repo\python\output\neural_recon\img_nif_log"
    def load_img_model(img_name):
        if os.path.exists(os.path.join(img_model_root_dir, img_name)):
            checkpoint_name = [item for item in os.listdir(os.path.join(img_model_root_dir, img_name)) if
                               item[-4:] == "ckpt"]
            assert len(checkpoint_name) == 1
            state_dict = torch.load(os.path.join(img_model_root_dir, img_name, checkpoint_name[0]))["state_dict"]
            fake_cfg = {
                "trainer": {
                    "learning_rate": 0,
                    "batch_size": 0,
                    "num_worker": 0,
                    "output": "output",
                },
                "dataset": {
                    "img_size": [600, 400],
                }
            }
            img_model = Phase1(fake_cfg, img1.img_path)
            img_model.load_state_dict(state_dict, strict=True)
            img_model.eval()
            return img_model
        else:
            print("cannot find model for img {}".format(img_name))
            raise

    img_model1 = load_img_model(img1.img_name)
    img_model2 = load_img_model(img2.img_name)

    optimize_single_segment_tensor(
        segment_3d_distances[changed_id_segment],
        lines1[id_segment],
        rgb1, rgb2,
        img1.intrinsic, img1.extrinsic,
        img2.intrinsic, img2.extrinsic,
        img_model1, img_model2
    )

    print("Calculate epipolar lines")


    def calculate_epipolar_line():
        segment = lines1[id_segment]

        F = extract_fundamental_from_projection(
            img2.extrinsic @ np.linalg.inv(img1.extrinsic),
            img1.intrinsic,
            img2.intrinsic,
        )

        epipolar_line1 = F @ to_homogeneous(segment[:2])
        epipolar_line2 = F @ to_homogeneous(segment[2:])
        query_point1 = np.array((
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 1),
        ))
        query_point2 = query_point1.copy()
        query_point1[:, 1] = -(epipolar_line1[0] * query_point1[:, 0] + epipolar_line1[2]) / epipolar_line1[1]
        query_point2[:, 1] = -(epipolar_line2[0] * query_point2[:, 0] + epipolar_line2[2]) / epipolar_line2[1]
        query_point1 = (query_point1 * shape).astype(np.int32)
        query_point2 = (query_point2 * shape).astype(np.int32)

        cv2.line(rgb1, (segment[:2] * shape).astype(np.int32), (segment[2:] * shape).astype(np.int32), (255, 0, 0),
                 thickness=10)
        cv2.circle(rgb1, (segment[:2] * shape).astype(np.int32), 20, (0, 0, 255), thickness=40)
        cv2.circle(rgb1, (segment[2:] * shape).astype(np.int32), 20, (0, 255, 0), thickness=40)
        cv2.line(rgb2, query_point1[0], query_point1[1], (0, 0, 255), thickness=20)
        cv2.line(rgb2, query_point2[0], query_point2[1], (0, 255, 0), thickness=20)
        cv2.imshow("1", np.concatenate((rgb1, rgb2), axis=0))
        cv2.waitKey()
        pass
    calculate_epipolar_line()
