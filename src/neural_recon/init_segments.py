import ray
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add

from shared.common_utils import to_homogeneous_mat_tensor, to_homogeneous_tensor
from src.neural_recon.optimize_segment import sample_img_prediction, sample_img


def sample_edge(num_per_edge_m2, cur_dir, start_point, max_num_points_per_edge = 2000):
    length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
    num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, max_num_points_per_edge)
    num_edge_points_ = num_edge_points.roll(1)
    num_edge_points_[0] = 0
    sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
        dim=0).repeat_interleave(num_edge_points)
    sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
    sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                          + start_point.repeat_interleave(num_edge_points, dim=0)
    return num_edge_points, sampled_edge_points

@ray.remote(num_gpus=1)
def compute_init_based_on_similarity(v_ray_c, v_intrinsic1, v_intrinsic2,
                                     v_extrinsic1, v_extrinsic2,
                                     v_img_model1, v_img_model2,
                                     v_rgb1, v_rgb2):
    with torch.no_grad():
        min_distance = 1
        max_distance = 300
        img_method = "img"

        num_task = v_ray_c.shape[0]
        result_distances = []
        for id_task in range(num_task):
            torch.cuda.empty_cache()

            v_ray_c1 = v_ray_c[id_task,0]
            v_ray_c2 = v_ray_c[id_task,1]

            dis1 = np.arange(min_distance, max_distance, 1)
            dis2 = np.arange(min_distance, max_distance, 1)

            dis1 = dis1.repeat(dis1.shape)
            dis2 = np.tile(dis2, dis2.shape)

            points1 = torch.tensor(v_ray_c1[None, :] * dis1[:, None], dtype=torch.float32).cuda()
            points2 = torch.tensor(v_ray_c2[None, :] * dis2[:, None], dtype=torch.float32).cuda()

            cur_dir = points2 - points1

            # Calculate the sample points on this edge in the camera coordinate
            num_edge_points, sampled_edge_points = sample_edge(100, cur_dir, points1, max_num_points_per_edge=1000)
            edge_index_1d = torch.arange(num_edge_points.shape[0],device=num_edge_points.device).repeat_interleave(num_edge_points)

            coordinates = sampled_edge_points

            # Calculate the projected 2d coordinates on both imgs
            intrinsic1 = torch.tensor(v_intrinsic1, dtype=torch.float32).cuda()
            intrinsic2 = torch.tensor(v_intrinsic2, dtype=torch.float32).cuda()
            extrinsic1 = torch.tensor(v_extrinsic1, dtype=torch.float32).cuda()
            extrinsic2 = torch.tensor(v_extrinsic2, dtype=torch.float32).cuda()

            coor_2d1 = (intrinsic1 @ coordinates.T).T
            coor_2d1 = coor_2d1[:, :2] / (coor_2d1[:, 2:3] + 1e-6)
            transformation = to_homogeneous_mat_tensor(intrinsic2) @ extrinsic2 @ torch.inverse(
                extrinsic1)
            coor_2d2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
            coor_2d2 = coor_2d2[:, :2] / (coor_2d2[:, 2:3] + 1e-6)
            valid_mask = torch.logical_and(coor_2d2 > 0, coor_2d2 < 1)
            valid_mask = torch.logical_and(valid_mask[:, 0], valid_mask[:, 1])
            valid_mask = scatter_add(valid_mask.to(torch.int32), edge_index_1d) == num_edge_points

            if valid_mask.sum() == 0:
                result_distances.append(-1)
                result_distances.append(-1)
                continue

            # Filter out useless projection
            dis1 = dis1[valid_mask.cpu().numpy()]
            dis2 = dis2[valid_mask.cpu().numpy()]
            coor_mask = valid_mask.repeat_interleave(num_edge_points)
            coor_2d1 = coor_2d1[coor_mask]
            coor_2d2 = coor_2d2[coor_mask]
            num_edge_points = num_edge_points[valid_mask]
            edge_index_1d = torch.arange(num_edge_points.shape[0],device=num_edge_points.device).repeat_interleave(num_edge_points)

            if img_method == "model":
                coor_2d1 = torch.clamp(coor_2d1, 0, 0.999999)
                sample_imgs1 = sample_img_prediction(v_img_model1, coor_2d1[None, :, :])[0]
            else:
                sample_imgs1 = sample_img(torch.tensor(v_rgb1,dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda(), coor_2d1[None, :, :])[0]
            if img_method == "model":
                coor_2d2 = torch.clamp(coor_2d2, 0, 0.999999)
                sample_imgs2 = sample_img_prediction(v_img_model2, coor_2d2[None, :, :])[0]
            else:
                sample_imgs2 = sample_img(torch.tensor(v_rgb2,dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda(), coor_2d2[None, :, :])[0]

            # 7. Similarity loss
            similarity_loss = ((sample_imgs1 - sample_imgs2)**2).mean(dim = 1)
            loss_per_edge = scatter_add(similarity_loss, dim=0, index=edge_index_1d)

            id_best = loss_per_edge.argmin()
            best_dis1 = dis1[id_best]
            best_dis2 = dis2[id_best]
            result_distances.append(best_dis1)
            result_distances.append(best_dis2)
    return result_distances


