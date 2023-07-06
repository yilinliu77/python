from math import ceil

import cv2
import numba
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

from shared.common_utils import normalize_vector, normalized_torch_img_to_numpy, padding, to_homogeneous, \
    to_homogeneous_vector, to_homogeneous_tensor, normalize_tensor, to_homogeneous_mat_tensor


def compute_initial_normal_based_on_camera(
        points_pos_3d_c: np.ndarray,
        v_graph: nx.Graph,
):
    point_index = []
    for id_face, face_ids in enumerate(v_graph.graph["faces"]):
        for id_segment in range(len(face_ids)):
            point_index.append(face_ids[id_segment])
            point_index.append(face_ids[(id_segment + 1) % len(face_ids)])
            point_index.append(id_face)
    point_index = np.asarray(point_index).reshape((-1, 3))
    s_c = points_pos_3d_c[point_index[:, 0]]
    e_c = points_pos_3d_c[point_index[:, 1]]
    c_c = (s_c + e_c) / 2
    center_to_origin = -normalize_vector(c_c)
    start_to_end = normalize_vector(s_c - e_c)
    up = normalize_vector(np.cross(center_to_origin, start_to_end))
    # up = normalize_vector(c_c)
    # normal = normalize_vector(np.cross(up, start_to_end))
    for id_edge, edge in enumerate(point_index):
        if "up_c" not in v_graph[edge[0]][edge[1]]:
            v_graph[edge[0]][edge[1]]["up_c"] = {edge[2]: up[id_edge]}
        else:
            v_graph[edge[0]][edge[1]]["up_c"].update({edge[2]: up[id_edge]})

    return


def sample_img(v_rgb, v_coor_2d):
    MAX_ITEM_PER_BATCH = 262144
    coor_2d = v_coor_2d
    results = []
    for i in range(ceil(coor_2d.shape[1] / MAX_ITEM_PER_BATCH)):
        sampled_pixels_flatten = torch.nn.functional.grid_sample(v_rgb,
            coor_2d[:,i * MAX_ITEM_PER_BATCH:min(coor_2d.shape[1], (i + 1) * MAX_ITEM_PER_BATCH)].unsqueeze(1) * 2 - 1,
                                                                 align_corners=True)[0,:,0]
        results.append(sampled_pixels_flatten)
    sampled_pixels = torch.cat(results,dim=1).permute(1,0)

    return sampled_pixels


def sample_img_prediction(v_model, v_coor_2d):
    MAX_ITEM_PER_BATCH = 262144
    coor_2d = v_coor_2d.reshape((-1, 2))
    results = []
    for i in range(ceil(coor_2d.shape[0] / MAX_ITEM_PER_BATCH)):
        sampled_pixels_flatten = v_model(
            coor_2d[i * MAX_ITEM_PER_BATCH:min(coor_2d.shape[0], (i + 1) * MAX_ITEM_PER_BATCH)])
        results.append(sampled_pixels_flatten)
    sampled_pixels = torch.cat(results, dim=0).reshape(v_coor_2d.shape[:2] + (3,))
    return sampled_pixels

def sample_img_prediction2(v_model, v_coor_2d, v_name):
    MAX_ITEM_PER_BATCH = 262144
    coor_2d = v_coor_2d.reshape((-1, 2))
    results = []
    for i in range(ceil(coor_2d.shape[0] / MAX_ITEM_PER_BATCH)):
        sampled_pixels_flatten = v_model(v_name,
            coor_2d[i * MAX_ITEM_PER_BATCH:min(coor_2d.shape[0], (i + 1) * MAX_ITEM_PER_BATCH)])
        results.append(sampled_pixels_flatten[1])
    sampled_pixels = torch.cat(results, dim=0).reshape(v_coor_2d.shape[:2] + (1,))
    return sampled_pixels

### Out of date

def compute_initial_normal(
        v_start_point_in_camera,
        v_end_point_in_camera,
        v_center_point_in_camera
):
    center_to_origin = -normalize_vector(v_center_point_in_camera[:3])
    start_to_end = normalize_vector(v_start_point_in_camera[:3] - v_end_point_in_camera[:3])
    up = np.cross(center_to_origin, start_to_end)
    normal = np.cross(up, start_to_end)
    return normalize_vector(start_to_end), normalize_vector(up), normalize_vector(normal)


def compute_initial_normal_based_on_pos(v_graph: nx.Graph):
    for id_face, face_ids in enumerate(tqdm(v_graph.graph["faces"])):
        for id_segment in range(len(face_ids)):
            id_start = face_ids[id_segment]
            id_end = face_ids[(id_segment + 1) % len(face_ids)]
            id_next = face_ids[(id_segment + 2) % len(face_ids)]

            cur_segment = v_graph.nodes[id_end]["ray_c"] * v_graph.nodes[id_end]["distance"] - v_graph.nodes[id_start][
                "ray_c"] * v_graph.nodes[id_start]["distance"]
            next_segment = v_graph.nodes[id_next]["ray_c"] * v_graph.nodes[id_next]["distance"] - v_graph.nodes[id_end][
                "ray_c"] * v_graph.nodes[id_end]["distance"]

            normal = np.cross(cur_segment, next_segment)
            sign_flag = normal.dot(np.array((0, 0, 1))) > 0
            if sign_flag:
                normal = -normal

            if "up_c" not in v_graph[id_start][id_end]:
                v_graph[id_start][id_end]["up_c"] = {id_face: normalize_vector(np.cross(normal, cur_segment))}
            else:
                v_graph[id_start][id_end]["up_c"].update({id_face: normalize_vector(np.cross(normal, cur_segment))})


# All coordinate are in camera space
def compute_roi(
        s,  # Start point
        e,  # end point
        c,  # center point
        v_line, v_up, v_length,
        v_intrinsic1
):
    device = s.device
    half_window_size_meter_horizontal = torch.norm(s - e) / 2  # m
    half_window_size_meter_vertical = v_length  # m
    half_window_size_step = 0.05

    # Compute extreme point
    point1 = s
    point2 = s + v_up * half_window_size_meter_vertical
    point3 = e + v_up * half_window_size_meter_vertical
    point4 = e
    roi = torch.stack((point1, point2, point3, point4), dim=0)
    roi_2d = torch.transpose(v_intrinsic1 @ torch.transpose(roi, 0, 1), 0, 1)
    roi_2d = roi_2d[:, :2] / roi_2d[:, 2:3]

    # Compute interpolated point
    num_horizontal = max(1, half_window_size_meter_horizontal.item() // half_window_size_step)
    num_vertical = max(1, half_window_size_meter_vertical.item() // half_window_size_step)
    dx = torch.arange(-num_horizontal, num_horizontal,
                      device=device) / num_horizontal * half_window_size_meter_horizontal
    dy = torch.arange(num_vertical, device=device) / num_vertical * half_window_size_meter_vertical

    dxdy = torch.stack(torch.meshgrid(dx, dy, indexing='xy'), dim=-1).to(device)
    interpolated_coordinates_camera = v_line[None, None, :] * dxdy[:, :, 0:1] + v_up[None, None, :] * dxdy[:, :,
                                                                                                      1:2] + c
    return roi, interpolated_coordinates_camera, roi_2d


def optimize_single_segment(
        v_seg_distance,  # Distances of the two endpoints
        v_seg_2d,  # Image coordinate of the two endpoints
        v_rgb1, v_rgb2,  # Images
        v_intrinsic1, v_extrinsic1,
        v_intrinsic2, v_extrinsic2,
        v_img_model1, v_img_model2
):
    v_is_debug = True
    if v_is_debug:
        print("Visualize the input segment")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        cv2.circle(line_img1, (v_seg_2d[0:2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (v_seg_2d[2:4] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.imshow("1", line_img1)
        cv2.waitKey()

    ray_c = np.transpose(np.linalg.inv(v_intrinsic1) @ np.transpose(np.insert(v_seg_2d.reshape((2, 2)), 2, 1, axis=1)))
    ray_c = ray_c / np.linalg.norm(ray_c, axis=1, keepdims=True)
    seg_3d_c = ray_c * v_seg_distance[:, np.newaxis]
    center_3d_c = (seg_3d_c[0] + seg_3d_c[1]) / 2
    v_along_line_c, v_up_c, v_normal_c = compute_initial_normal(
        seg_3d_c[0], seg_3d_c[1], center_3d_c
    )

    roi_bound, roi_coor, roi_2d = compute_roi(
        seg_3d_c[0], seg_3d_c[1], center_3d_c,
        v_along_line_c, v_up_c, v_normal_c,
        v_intrinsic1
    )

    # Visualize
    if v_is_debug:
        print("Visualize the calculated roi")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        line_img1 = cv2.polylines(line_img1, [(roi_2d * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                  (0, 255, 0),
                                  thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(line_img1, (roi_2d[0] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d[1] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d[2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d[3] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.imshow("1", line_img1)
        cv2.waitKey()

    roi_coor_shape = roi_coor.shape
    roi_coor_2d = np.transpose(v_intrinsic1 @ np.transpose(roi_coor.reshape((-1, 3))))
    roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
    roi_coor_2d = roi_coor_2d.reshape(roi_coor_shape[:2] + (2,))

    viz_sampled_img = sample_img(v_rgb1, roi_coor_2d)

    if v_is_debug:
        print("Visualize the extracted region")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        for item in roi_coor_2d.reshape((-1, 2)):
            cv2.circle(line_img1, (item * shape).astype(np.int32), 2, (0, 0, 255), 2)

        sample_shape = viz_sampled_img.shape[:2]
        resized_shape = (shape[0], shape[0] // sample_shape[1] * sample_shape[0])
        cv2.imshow("1", np.concatenate((
            line_img1,
            padding(
                cv2.resize(viz_sampled_img, resized_shape),
                shape[1], shape[0])
        ), axis=1))
        cv2.waitKey()

    transformation = to_homogeneous(v_intrinsic2) @ v_extrinsic2 @ np.linalg.inv(v_extrinsic1)
    roi_coor_2d_img2 = np.transpose(transformation @ np.transpose(to_homogeneous_vector(roi_coor.reshape((-1, 3)))))
    roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
    roi_coor_2d_img2 = roi_coor_2d_img2.reshape(roi_coor_shape[:2] + (2,))
    viz_sampled_img2 = sample_img(v_rgb2, roi_coor_2d_img2)

    if v_is_debug:
        print("Visualize the extracted region")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        for item in roi_coor_2d.reshape((-1, 2)):
            cv2.circle(line_img1, (item * shape).astype(np.int32), 2, (0, 0, 255), 2)
        sample_shape = viz_sampled_img.shape[:2]
        resized_shape = (shape[0], shape[0] // sample_shape[1] * sample_shape[0])
        img1 = np.concatenate((
            line_img1,
            padding(
                cv2.resize(viz_sampled_img, resized_shape),
                shape[1], shape[0])
        ), axis=1)

        line_img2 = v_rgb2.copy()
        shape = v_rgb2.shape[:2][::-1]
        for item in roi_coor_2d_img2.reshape((-1, 2)):
            cv2.circle(line_img2, (item * shape).astype(np.int32), 2, (0, 0, 255), 2)
        sample_shape = viz_sampled_img2.shape[:2]
        resized_shape = (shape[0], shape[0] // sample_shape[1] * sample_shape[0])
        img2 = np.concatenate((
            line_img2,
            padding(
                cv2.resize(viz_sampled_img2, resized_shape),
                shape[1], shape[0])
        ), axis=1)

        cv2.imshow("1", np.concatenate((img1, img2), axis=0))
        cv2.waitKey()
    pass


def optimize_single_segment_tensor(
        v_seg_distance,  # Distances of the two endpoints
        v_seg_2d,  # Image coordinate of the two endpoints
        v_rgb1, v_rgb2,  # Images
        v_intrinsic1, v_extrinsic1,
        v_intrinsic2, v_extrinsic2,
        v_img_model1, v_img_model2
):
    v_is_debug = True
    if v_is_debug:
        print("Visualize the input segment")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        cv2.circle(line_img1, (v_seg_2d[0:2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (v_seg_2d[2:4] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.imshow("1", line_img1)
        cv2.waitKey()

    seg_2d = torch.as_tensor(v_seg_2d).float()
    seg_distance = torch.as_tensor(v_seg_distance).float()

    intrinsic1 = torch.as_tensor(v_intrinsic1).float()
    intrinsic2 = torch.as_tensor(v_intrinsic2).float()
    extrinsic1 = torch.as_tensor(v_extrinsic1).float()
    extrinsic2 = torch.as_tensor(v_extrinsic2).float()

    ray_c = torch.transpose(
        torch.inverse(intrinsic1) @ torch.transpose(to_homogeneous_tensor(seg_2d.reshape((2, 2))), 0, 1), 0, 1)
    ray_c = ray_c / torch.norm(ray_c, dim=1, keepdim=True)
    seg_3d_c = ray_c * seg_distance[:, None]
    center_3d_c = (seg_3d_c[0] + seg_3d_c[1]) / 2
    v_along_line_c, v_up_c, v_normal_c = compute_initial_normal(
        seg_3d_c[0], seg_3d_c[1], center_3d_c
    )

    roi_bound, roi_coor, roi_2d = compute_roi(
        seg_3d_c[0], seg_3d_c[1], center_3d_c,
        v_along_line_c, v_up_c, v_normal_c,
        intrinsic1
    )

    # Visualize
    if v_is_debug:
        print("Visualize the calculated roi")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]
        roi_2d_numpy = roi_2d.detach().cpu().numpy()
        line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                  (0, 255, 0),
                                  thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(line_img1, (roi_2d_numpy[0] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d_numpy[1] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d_numpy[2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.circle(line_img1, (roi_2d_numpy[3] * shape).astype(np.int32), 10, (0, 0, 255), 10)
        cv2.imshow("1", line_img1)
        cv2.waitKey()

    roi_coor_shape = roi_coor.shape
    roi_coor_2d = torch.transpose(intrinsic1 @ torch.transpose(roi_coor.reshape((-1, 3)), 0, 1), 0, 1)
    roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
    roi_coor_2d = roi_coor_2d.reshape(roi_coor_shape[:2] + (2,))

    sampled_img = sample_img_prediction(v_img_model1, roi_coor_2d)

    transformation = to_homogeneous_mat_tensor(intrinsic2) @ extrinsic2 @ torch.inverse(extrinsic1)
    roi_coor_2d_img2 = torch.transpose(
        transformation @ torch.transpose(to_homogeneous_tensor(roi_coor.reshape((-1, 3))), 0, 1), 0, 1)
    roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
    roi_coor_2d_img2 = roi_coor_2d_img2.reshape(roi_coor_shape[:2] + (2,))
    viz_sampled_img2 = sample_img_prediction(v_img_model2, roi_coor_2d_img2)

    if v_is_debug:
        print("Visualize the extracted region")
        line_img1 = v_rgb1.copy()
        shape = v_rgb1.shape[:2][::-1]

        roi_coor_2d1_numpy = roi_coor_2d.detach().cpu().numpy()
        sampled_img1_numpy = normalized_torch_img_to_numpy(sampled_img.permute(2, 0, 1))
        roi_coor_2d2_numpy = roi_coor_2d_img2.detach().cpu().numpy()
        sampled_img2_numpy = normalized_torch_img_to_numpy(viz_sampled_img2.permute(2, 0, 1))

        for item in roi_coor_2d1_numpy.reshape((-1, 2)):
            cv2.circle(line_img1, (item * shape).astype(np.int32), 2, (0, 0, 255), 2)
        sample_shape = sampled_img1_numpy.shape[:2]
        resized_shape = (shape[0], shape[0] // sample_shape[1] * sample_shape[0])
        img1 = np.concatenate((
            line_img1,
            padding(
                cv2.resize(sampled_img1_numpy, resized_shape),
                shape[1], shape[0])
        ), axis=1)

        line_img2 = v_rgb2.copy()
        shape = v_rgb2.shape[:2][::-1]
        for item in roi_coor_2d2_numpy.reshape((-1, 2)):
            cv2.circle(line_img2, (item * shape).astype(np.int32), 2, (0, 0, 255), 2)
        sample_shape = sampled_img2_numpy.shape[:2]
        resized_shape = (shape[0], shape[0] // sample_shape[1] * sample_shape[0])
        img2 = np.concatenate((
            line_img2,
            padding(
                cv2.resize(sampled_img2_numpy, resized_shape),
                shape[1], shape[0])
        ), axis=1)

        cv2.imshow("1", np.concatenate((img1, img2), axis=0))
        cv2.waitKey()

    torch.nn.functional.mse_loss(sampled_img, viz_sampled_img2)

    pass
