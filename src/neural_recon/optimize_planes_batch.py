import os
import math
import pickle
from collections import deque

import cv2
import networkx as nx
import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_sum

from shared.common_utils import normalize_tensor, to_homogeneous_tensor
from src.neural_recon.Visualizer import Visualizer
from src.neural_recon.collision_checker import Collision_checker
from src.neural_recon.geometric_util import angles_to_vectors, compute_plane_abcd, intersection_of_ray_and_all_plane, \
    intersection_of_ray_and_plane
from src.neural_recon.io_utils import generate_random_color, save_plane
from src.neural_recon.loss_utils import (compute_regularization, Glue_loss_computer, Bilateral_ncc_computer,
                                         Edge_loss_computer, Regularization_loss_computer, Mutex_loss_computer,
                                         dilate_edge, get_projections, get_projections_batch)
from src.neural_recon.sample_utils import sample_new_planes, sample_triangles

import os
import faiss

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CosineAnnealingScaleFactorScheduler:
    def __init__(self, initial_sf=3.0, initial_sf_mult=0.99, T_0=12, T_mult=1.01, min_sf=0.001):
        """Implements Cosine Annealing with Restarts learning rate scheduler.
        Args:
            initial_sf (float): Initial scale factor.
            initial_sf_mult (float): Scale factor multiplier of initial_sf after a restart.
            T_0 (int): Number of iterations for the first restart.
            T_mult (int): A factor increases T_0 after a restart. Default: 1
            min_sf (float): Minimum scale factor. Default: 0.001
        """
        self.initial_sf = initial_sf
        self.initial_sf_mult = initial_sf_mult
        self.init_sf_cur_cycle = initial_sf
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_sf = min_sf

        self.cur_cycle = 0
        self.cur_iter = 0
        self.sf_history = []

    def get_sf(self):
        if self.cur_iter != 0 and self.cur_iter % self.T_0 == 0:
            self.cur_iter = 0
            self.cur_cycle += 1

            if self.cur_cycle != 0 and self.cur_cycle % 2 == 0:
                self.initial_sf *= self.initial_sf_mult

            self.init_sf_cur_cycle = self.initial_sf * np.clip(np.random.standard_cauchy(), 0.001, 3.0)
            # self.init_sf_cur_cycle = self.initial_sf * np.random.uniform(0, 3)

            new_T_0 = int(self.T_0 * self.T_mult)
            self.T_0 = new_T_0 if new_T_0 % 2 == 0 else new_T_0 + 1

        sf = self.min_sf + (self.init_sf_cur_cycle - self.min_sf) * (
                1 + math.cos(math.pi * (self.cur_iter % self.T_0) / self.T_0)) / 2

        self.sf_history.append(sf)
        self.cur_iter += 1
        return sf


def optimize_planes_batch(initialized_planes, v_rays_c, v_centroid_rays_c, src_faces_centroid_rays_c,
                          dual_graph, imgs, transformation, extrinsic_ref_cam, intrinsic, v_c1_2_c2_list,
                          v_log_root, optimized_abcd_list=None):
    # Prepare some data
    device = initialized_planes.device
    patch_num = len(initialized_planes)

    patches_list = dual_graph.nodes  # each patch = id_vertexes
    patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]

    dilated_gradients_list = []
    for i in range(0, len(imgs)):
        dy, dx = torch.gradient(imgs[i])
        gradients = torch.stack((dx, dy), dim=-1)
        dilated_gradients = torch.from_numpy(dilate_edge(gradients)).to(device)
        # dilated_gradients_list.append(dilated_gradients)
        dilated_gradients_list.append(gradients)
    dilated_gradients_list = torch.stack(dilated_gradients_list, dim=0)

    # sub faces of each patch
    # sub_faces = [dual_graph.nodes[i]['sub_faces'] for i in range(patch_num)]
    # sub_faces_id = [dual_graph.nodes[i]['sub_faces_id'] for i in range(patch_num)]
    # for i in range(patch_num):
    #     sub_faces_c = dual_graph.nodes[i]['sub_faces']
    #     for sub_face in sub_faces_c:
    #         pass

    # optimization loop control variables
    cur_iter = [0] * patch_num
    best_loss = [None] * patch_num
    delta = [None] * patch_num
    MAX_ITER = 1001
    MAX_TOLERENCE = 100
    num_tolerence = [MAX_TOLERENCE] * patch_num
    num_plane_sample = 100

    if optimized_abcd_list is None:
        optimized_abcd_list = initialized_planes.clone()
    else:
        MAX_ITER = 1

    tri_colors = [generate_random_color() for _ in range(100)]
    sample_g = torch.Generator(device)
    sample_g.manual_seed(0)
    end_flag = [0] * patch_num

    # scale_factor adapter1
    # init_scale_factor = 3.0
    # gamma = 0.8
    # milestones = [150, 150, 250, 250, 100, 100, 100]
    # scale_factors = [init_scale_factor * gamma ** i for i in range(len(milestones))]
    # def get_scale_factor1(cur_iter):
    #     scale_factor = scale_factors[-1]
    #     for i in range(len(milestones)):
    #         if cur_iter < milestones[i]:
    #             scale_factor = scale_factors[i]
    #             break
    #     # return scale_factor
    #     return np.clip(abs(np.random.standard_cauchy() * scale_factor), 0.1, 3.0)

    # scale_factor adapter2
    CASF = CosineAnnealingScaleFactorScheduler()

    # loss weight
    ncc_loss_weight = 1
    edge_loss_weight = 10
    reg_loss_weight = 0
    glue_loss_weight = 1
    regularization_loss_weight = 0
    mutex_loss_weight = 0
    edge_loss_computer = Edge_loss_computer(v_sample_density=0.001)
    glue_loss_computer = Glue_loss_computer(dual_graph, v_rays_c)
    regularization_loss_computer = Regularization_loss_computer(dual_graph)
    mutex_loss_computer = Mutex_loss_computer()
    ncc_loss_computer = Bilateral_ncc_computer(
            v_enable_spatial_weights=True,
            v_enable_color_weights=True,
            v_window_size=7
            )
    collision_checker = Collision_checker()

    id_neighbour_patches = [list(dual_graph[patch_id].keys()) for patch_id in dual_graph.nodes]

    # Start to optimize: patch based optimization
    # 1. partition the patches into several groups using the nx.greedy_color(dual_graph)
    is_clusters = False
    if is_clusters:
        patch_color = nx.greedy_color(dual_graph)
        groups = {}
        for key, value in patch_color.items():
            if value not in groups:
                groups[value] = [key]
            else:
                groups[value].append(key)
        groups = list(groups.values())
    else:
        groups = [list(range(patch_num))]
    group_idx = -1
    count = 0

    # debug_id_list = torch.tensor([652,813,972,1029,1209,1492,1500,1600,1700]).to(device)
    debug_id_list = torch.arange(patch_num).to(device)

    # sort the patches by the area of neighbour patches
    # sorted_patch_id_list = sorted(dual_graph.nodes, key=lambda x: len(dual_graph[x]), reverse=True)
    area_list = []
    for patch_id in range(patch_num):
        local_vertex_pos = intersection_of_ray_and_all_plane(optimized_abcd_list[patch_id].unsqueeze(0),
                                                             v_rays_c[patch_vertexes_id[patch_id]])  # 100 * n * 3
        # 2. Construct the triangles
        num_vertex = len(patch_vertexes_id[patch_id])
        edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
        local_edge_pos = local_vertex_pos[:, edges_idx]
        local_centroid = intersection_of_ray_and_all_plane(optimized_abcd_list[patch_id].unsqueeze(0),
                                                           v_centroid_rays_c[patch_id].unsqueeze(0))[:, 0]
        triangles_pos = torch.cat((local_edge_pos, local_centroid[:, None, None, :].tile(1, num_vertex, 1, 1)),
                                  dim=2)
        triangles_pos = triangles_pos.view(-1, 3, 3)  # (1*num_tri,3,3)

        d1 = triangles_pos[:, 1, :] - triangles_pos[:, 0, :]
        d2 = triangles_pos[:, 2, :] - triangles_pos[:, 1, :]
        area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2
        area_list.append(area.sum().item())

    debug_id_list = sorted(debug_id_list, key=lambda x: area_list[x], reverse=True)
    debug_id_list = torch.tensor(debug_id_list).to(device)

    visualizer = Visualizer(
            patch_num, v_log_root,
            imgs,
            intrinsic,
            extrinsic_ref_cam,
            transformation,
            debug_id_list
            )
    visualizer.save_planes("init_plane.ply", initialized_planes, v_rays_c, patch_vertexes_id)

    # vis the choosen debug patch
    visualizer.viz_results(
            -1, initialized_planes, v_rays_c, patch_vertexes_id, debug_id_list
            )

    while True:
        # if count % 50 == 0:
        group_idx = (group_idx + 1) % len(groups)
        visualizer.start_iter()
        # 1. Sample new hypothesis of all patch from 1) propagation 2) random perturbation
        # patch_id_list = torch.tensor(groups[group_idx]).to(device)
        patch_id_list = debug_id_list.clone()
        patch_num = len(patch_id_list)
        samples_depth, samples_angle = sample_new_planes(optimized_abcd_list,
                                                         v_centroid_rays_c,
                                                         id_neighbour_patches,
                                                         patch_id_list,
                                                         CASF.get_sf(),
                                                         v_random_g=sample_g)
        samples_abc = angles_to_vectors(samples_angle)  # n * 100 * 3
        samples_intersection, samples_abcd = compute_plane_abcd(v_centroid_rays_c[patch_id_list], samples_depth,
                                                                samples_abc)

        visualizer.update_timer("Sample")
        visualizer.update_sample_plane(patch_id_list, samples_abcd, v_rays_c, patch_vertexes_id, cur_iter[0])
        visualizer.update_timer("SampleVis")

        if sum(end_flag) == patch_num:
            break

        # if min(cur_iter) > 50:
        #     glue_loss_weight = 1

        # for v_patch_id in range(patch_num):
        for idx in range(len(patch_id_list)):
            v_patch_id = patch_id_list[idx]

            if end_flag[v_patch_id] == 1:
                continue

            # 1. compute some share variables of this patch
            v_img1 = imgs[0]
            dilated_gradients1 = dilated_gradients_list[0]
            num_vertex = len(patch_vertexes_id[v_patch_id])
            local_plane_parameter = samples_abcd[idx]
            local_vertex_pos = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                                 v_rays_c[patch_vertexes_id[v_patch_id]])  # 100 * n * 3
            visualizer.update_timer("Construct_p1")
            edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
            local_edge_pos = local_vertex_pos[:, edges_idx]
            local_centroid = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                               v_centroid_rays_c[v_patch_id].unsqueeze(0))[:, 0]

            visualizer.update_timer("Construct_p2")

            # 2. Construct the triangles
            sub_faces_id_c = dual_graph.nodes[int(v_patch_id)]['sub_faces_id']
            sub_faces_c = dual_graph.nodes[int(v_patch_id)]['sub_faces']
            triangles_pos_list = []
            triangles_num_per_face = []
            for sub_face_id, sub_face_vertices_id in zip(sub_faces_id_c, sub_faces_c):
                vertex_pos_sub_face = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                                        v_rays_c[sub_face_vertices_id])  # 100 * n * 3
                num_vertex_sf = len(sub_face_vertices_id)
                triangles_num_per_face.append(num_vertex_sf)
                indices = torch.arange(num_vertex_sf)
                edges_idx = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1).tolist()
                visualizer.update_timer("Construct_p3")
                edge_pos_sf = vertex_pos_sub_face[:, edges_idx]
                centroid_ray_sf = src_faces_centroid_rays_c[sub_face_id]
                centroid_sf = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                                centroid_ray_sf.unsqueeze(0))[:, 0]
                triangles_pos = torch.cat((edge_pos_sf, centroid_sf[:, None, None, :].tile(1, num_vertex_sf, 1, 1)),
                                          dim=2)
                triangles_pos_list.append(triangles_pos)  # (100,num_tri,3,3)
                visualizer.update_timer("Construct_p4")

            triangles_pos_per_sample = torch.cat(triangles_pos_list, dim=1)  # (100,num_tri,3,3)
            triangles_pos = triangles_pos_per_sample.view(-1, 3, 3)  # (100*num_tri,3,3)
            visualizer.update_timer("Construct")

            # 3. Sample points in this plane
            num_sample_points_per_tri, sample_points_on_face_src = sample_triangles(2500,
                                                                                    triangles_pos[:, 0, :],
                                                                                    triangles_pos[:, 1, :],
                                                                                    triangles_pos[:, 2, :],
                                                                                    v_sample_edge=False)

            triangle_normal = normalize_tensor(torch.cross(triangles_pos[:, 0, :] - triangles_pos[:, 1, :],
                                                           triangles_pos[:, 1, :] - triangles_pos[:, 2, :]))

            triangle_normal_src = triangle_normal.repeat_interleave(num_sample_points_per_tri, dim=0)
            points_to_tri_src = torch.arange(num_sample_points_per_tri.shape[0]).to(device).repeat_interleave(
                    num_sample_points_per_tri)

            visualizer.update_timer("Sample points")

            # Number of sample points on each face
            num_sp = sample_points_on_face_src.shape[0]
            # Number of source images
            num_source = transformation.shape[0]

            # 1. Check collision
            # we need to check collision for each source image in the reference image coordinate
            # 1. get camera coordinate of the source image in the coordinate of the reference image
            # 2. trans the ray's origin to the source image camera pos in the reference image coordinate
            camera_coor_in_c1 = torch.linalg.inv(torch.stack(v_c1_2_c2_list))[:, :3, -1]
            ray_dir = sample_points_on_face_src[None, :] - camera_coor_in_c1[:, None]
            collision_flag = collision_checker.check_ray(
                    v_patch_id,
                    camera_coor_in_c1[:, None].tile(1, num_sp, 1).reshape(-1, 3),
                    ray_dir.reshape(-1, 3),
                    batch_size=1000000)
            collision_flag = collision_flag.reshape(-1, num_sp)
            remain_flag = torch.logical_not(collision_flag)
            # Skip this src_img if all points are in collision
            valid_img_flag = torch.any(remain_flag, dim=1)

            visualizer.update_timer("Collision")

            # 2. Compute loss
            # 1) NCC loss
            points_ncc = ncc_loss_computer.compute_batch(
                    sample_points_on_face_src, triangle_normal_src,
                    intrinsic, transformation, v_img1, imgs[1:], batch_size=1000000)
            # Set the loss of invisible points to 0,
            # we do not consider this point when calculating the loss of this face
            points_ncc[~remain_flag] = 0
            visualizer.update_timer("NCC1")
            points_to_tri = points_to_tri_src[None, :].tile(num_source, 1)
            triangles_ncc = scatter_sum(points_ncc, points_to_tri, dim=1)
            triangles_ncc /= scatter_sum(remain_flag.to(torch.long), points_to_tri, dim=1)
            triangles_ncc = triangles_ncc.view(num_source, num_plane_sample, -1)
            triangle_weights = num_sample_points_per_tri.view(num_plane_sample, -1)

            triangle_weights = triangle_weights / (triangle_weights.sum(dim=-1, keepdim=True) + 1e-6)
            ncc_loss = (triangles_ncc * triangle_weights).nanmean(dim=-1)

            visualizer.update_timer("NCC2")

            # 2) Edge loss
            # v_c1_2_c2 = v_c1_2_c2_list[v_patch_id]
            # origin_c2 = torch.linalg.inv(v_c1_2_c2)[:3, -1].tile(local_edge_pos.reshape(-1, 3).shape[0], 1)
            # collision_flag_edge = collision_checker.check_ray(v_patch_id,
            #                                                   origin_c2,
            #                                                   local_edge_pos.reshape(-1, 3) - origin_c2)
            # collision_flag_edge = (collision_flag_edge.view(-1,2).sum(dim=1) == 2)  # 100 * num_edge
            # local_edge_pos[~collision_flag_edge.view(100,-1)]

            edge_loss, edge_outer_mask, num_samples_per_edge = edge_loss_computer.compute_batch(
                    local_edge_pos.reshape(-1, 2, 3),
                    intrinsic,
                    transformation,
                    dilated_gradients_list[0],
                    dilated_gradients_list[1:],
                    imgs[1:],
                    v_num_hypothesis=local_edge_pos.shape[0])
            edge_loss[~edge_outer_mask] = torch.inf

            edge_loss = edge_loss.view(num_source, num_plane_sample, -1)
            edge_loss = torch.nanmean(edge_loss, dim=2)

            visualizer.update_timer("Edge")

            # 3) Regularization loss
            # reg_loss = compute_regularization_batch(
            #     triangles_pos.view(num_plane_sample, -1, 3, 3),
            #     intrinsic,
            #     transformation)

            # 4) Glue loss
            glue_loss = glue_loss_computer.compute(v_patch_id, optimized_abcd_list, local_vertex_pos)
            visualizer.update_timer("Glue")

            # Final loss
            final_loss = ncc_loss * ncc_loss_weight \
                         + edge_loss * edge_loss_weight \
                         + glue_loss * glue_loss_weight

            final_loss[torch.logical_not(valid_img_flag)] = torch.inf
            ncc_loss[torch.logical_not(valid_img_flag)] = torch.inf
            edge_loss[torch.logical_not(valid_img_flag)] = torch.inf

            # the loss == inf: 1. when all the sample points of a sample are collided
            #                  2. ncc_loss == inf or edge_loss == inf
            final_loss = torch.where(final_loss.abs() == torch.inf, torch.nan, final_loss)
            ncc_loss_sum = torch.where(ncc_loss.abs() == torch.inf, torch.nan, ncc_loss)
            edge_loss_sum = torch.where(edge_loss.abs() == torch.inf, torch.nan, edge_loss)

            assert (final_loss == torch.inf).sum() == 0
            # filter based on the loss of the first image
            median = torch.nanmedian(final_loss[:, 0], dim=0)[0]
            outlier_mask = final_loss[:, 0] > median * 3
            assert outlier_mask.sum().item() != outlier_mask.shape[0]
            final_loss[outlier_mask] = torch.nan

            final_loss_sum = final_loss.nanmean(dim=0)
            # ncc_loss_sum = ncc_loss_sum.nanmean(dim=0)
            # edge_loss_sum = edge_loss_sum.nanmean(dim=0)

            visualizer.update_timer("Loss")

            # 6. Select the best and update `optimized_abcd_list`
            # all the sample of all the patch are collided
            # each sample should have at least 2 valid loss
            valid_loss_num = (~torch.isnan(final_loss)).sum(dim=0)
            final_loss_sum[torch.isnan(final_loss_sum)] = torch.inf

            id_best = torch.argmin(final_loss_sum, dim=-1)

            if best_loss[v_patch_id] is not None and valid_loss_num[id_best] < 2:
                id_best *= 0

            # if best_loss[v_patch_id] is not None and final_loss[id_best] < best_loss[v_patch_id]:
            optimized_abcd_list[v_patch_id] = samples_abcd[idx][id_best]

            # 7. Visualize
            visualizer.viz_patch_2d(
                    v_patch_id,
                    cur_iter[v_patch_id],

                    samples_abcd[idx],
                    sample_points_on_face_src,
                    num_sample_points_per_tri,

                    remain_flag,
                    triangles_pos_per_sample,
                    local_edge_pos,
                    local_centroid,

                    final_loss,
                    final_loss_sum,
                    ncc_loss,
                    edge_loss,
                    id_best,
                    )

            # 8. Control the end of the optimization
            cur_loss = final_loss_sum[id_best].cpu().item()
            if best_loss[v_patch_id] is not None:
                delta[v_patch_id] = cur_loss - best_loss[v_patch_id]
                print("Patch/Iter: {:3d}, {:4d}; Loss: {:.4f} Best: {:.4f} Delta: {:.4f} Id_Best: {:3d}".format(
                        v_patch_id,
                        cur_iter[v_patch_id] + 1,
                        cur_loss,
                        best_loss[v_patch_id],
                        delta[v_patch_id],
                        id_best.cpu().item())
                        )
                if abs(delta[v_patch_id]) < 1e-3:
                    num_tolerence[v_patch_id] -= 1
                else:
                    num_tolerence[v_patch_id] = MAX_TOLERENCE
                    best_loss[v_patch_id] = cur_loss

                if num_tolerence[v_patch_id] <= 0:  # or best_loss[v_patch_id] < 0.000001
                    end_flag[v_patch_id] = 1
            else:
                best_loss[v_patch_id] = cur_loss

            cur_iter[v_patch_id] += 1

            if cur_iter[v_patch_id] >= MAX_ITER:
                end_flag[v_patch_id] = 1

            visualizer.update_timer("Update1")

        # Update triangles
        final_triangles_list = []
        tri_num_each_patch = []
        for v_patch_id in range(patch_num):
            sub_faces_id_c = dual_graph.nodes[int(v_patch_id)]['sub_faces_id']
            sub_faces_c = dual_graph.nodes[int(v_patch_id)]['sub_faces']
            triangles_pos_list = []
            triangles_num_per_face = []
            for sub_face_id, sub_face_vertices_id in zip(sub_faces_id_c, sub_faces_c):
                vertex_pos_sub_face = intersection_of_ray_and_all_plane(optimized_abcd_list[v_patch_id:v_patch_id + 1],
                                                                        v_rays_c[sub_face_vertices_id])  # 100 * n * 3
                num_vertex_sf = len(sub_face_vertices_id)
                triangles_num_per_face.append(num_vertex_sf)
                indices = torch.arange(num_vertex_sf)
                edges_idx = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1).tolist()
                edge_pos_sf = vertex_pos_sub_face[:, edges_idx]
                centroid_ray_sf = src_faces_centroid_rays_c[sub_face_id]
                centroid_sf = intersection_of_ray_and_all_plane(optimized_abcd_list[v_patch_id:v_patch_id + 1],
                                                                centroid_ray_sf.unsqueeze(0))[:, 0]
                triangles_pos = torch.cat((edge_pos_sf, centroid_sf[:, None, None, :].tile(1, num_vertex_sf, 1, 1)),
                                          dim=2)
                triangles_pos_list.append(triangles_pos)  # (1,num_tri,3,3)

            triangles_pos_per_sample = torch.cat(triangles_pos_list, dim=1)  # (1,num_tri,3,3)
            tri_num_each_patch.append(triangles_pos_per_sample.shape[1])
            final_triangles_list.append(triangles_pos_per_sample[0])

        final_triangles = torch.cat(final_triangles_list)
        tri_to_patch = torch.arange(patch_num).repeat_interleave(torch.from_numpy(np.array(tri_num_each_patch))).to(
                device)
        collision_checker.clear()

        collision_checker.add_triangles(final_triangles, tri_to_patch)
        visualizer.update_timer("Update2")
        visualizer.viz_results(
                max(cur_iter),
                optimized_abcd_list, v_rays_c, patch_vertexes_id, debug_id_list
                )
        count += 1
        continue

    # save optimized_abcd_list
    save_plane(optimized_abcd_list, v_rays_c, patch_vertexes_id,
               file_path=os.path.join(v_log_root, "optimized.ply"))

    # save the loss of each patch
    for patch_id, loss in enumerate(best_loss):
        dual_graph.nodes[patch_id]['loss'] = loss

    with open("output/optimized_abcd_list.pkl", "wb") as f:
        pickle.dump(optimized_abcd_list, f)

    return optimized_abcd_list


def global_assemble(optimized_abcd_list_v,
                    transformation,
                    intrinsic,
                    v_c1_2_c2,
                    v_log_root
                    ):
    pass


def local_assemble(v_planes, v_rays_c, v_centroid_rays_c, dual_graph,
                   imgs, dilated_gradients1, dilated_gradients2,
                   transformation,
                   intrinsic,
                   v_c1_2_c2,
                   v_log_root
                   ):
    # 1. Prepare variables
    device = v_planes.device
    patch_num = len(v_planes)
    patches_list = dual_graph.nodes  # each patch = id_vertexes
    patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]

    num_plane_sample = 1000
    img_src_id = 2

    v_img1 = imgs[0]
    v_img2 = imgs[img_src_id + 1]

    tri_colors = [generate_random_color() for _ in range(100)]
    sample_g = torch.Generator(device)

    ncc_loss_weight = 1
    edge_loss_weight = 10
    reg_loss_weight = 1
    glue_loss_weight = 1
    regularization_loss_weight = 0
    edge_loss_computer = Edge_loss_computer(v_sample_density=0.001)
    glue_loss_computer = Glue_loss_computer(dual_graph, v_rays_c)
    regularization_loss_computer = Regularization_loss_computer(dual_graph)
    ncc_loss_computer = Bilateral_ncc_computer(
            v_enable_spatial_weights=True,
            v_enable_color_weights=True,
            v_window_size=7
            )
    collision_checker = Collision_checker()

    # find adj patch of each patch
    # adj_patch_list = []
    # adj_edge_list = []
    # for patch_id in range(len(dual_graph)):
    #     adj_patch_list.append(list(dual_graph[patch_id].keys()))
    #     adj_edge_list.append([item['adjacent_vertices'] for item in list(dual_graph[patch_id].values())])

    def init_merged_patch(patch1, patch2, shared_edge):
        shared_edge_p1_idx = [patch1.index(shared_edge[0]), patch1.index(shared_edge[1])]
        shared_edge_p2_idx = [patch2.index(shared_edge[0]), patch2.index(shared_edge[1])]
        merged_patch_ = patch1[shared_edge_p1_idx[1]:] + patch1[:shared_edge_p1_idx[1]]
        if shared_edge_p2_idx[1] < shared_edge_p2_idx[0]:
            shared_edge_p2_idx = shared_edge_p2_idx[::-1]
        merged_patch_ += patch2[shared_edge_p2_idx[1] + 1:] + patch2[:shared_edge_p2_idx[0]]
        return merged_patch_

    # 2. Choose initial patch
    init_patch_id = 2

    # eval src plane parameter of a patch
    def vis_megred_patch(merged_patch_id_list, merged_abcd, cur_iter_=0, text=None):
        ref_img = v_img1.cpu().numpy() * 255
        src_imgs = v_img2.cpu().numpy() * 255
        img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        img21 = cv2.cvtColor(src_imgs, cv2.COLOR_GRAY2BGR)
        img22 = cv2.cvtColor(src_imgs, cv2.COLOR_GRAY2BGR)

        def draw_sample(v_patch_id, local_plane_parameter, img11_, img21_):
            # 2. Get some variables
            num_plane_sample_ = local_plane_parameter.shape[0]
            num_vertex = len(patch_vertexes_id[v_patch_id])
            local_vertex_pos = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                                 v_rays_c[patch_vertexes_id[v_patch_id]])  # 100 * n * 3
            # 3. Construct the triangles
            edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
            local_edge_pos = local_vertex_pos[:, edges_idx]
            local_centroid = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                               v_centroid_rays_c[v_patch_id:v_patch_id + 1])[:, 0]
            triangles_pos = torch.cat((local_edge_pos, local_centroid[:, None, None, :].tile(1, num_vertex, 1, 1)),
                                      dim=2)
            triangles_pos = triangles_pos.view(-1, 3, 3)  # (100*num_tri,3,3)

            # 4. Sample points in this plane
            num_sample_points, sample_points_on_face = sample_triangles(100, triangles_pos[:, 0, :],
                                                                        triangles_pos[:, 1, :],
                                                                        triangles_pos[:, 2, :],
                                                                        v_sample_edge=False
                                                                        )

            # vis
            sample_points_c = sample_points_on_face
            all_points_c = local_vertex_pos[0].view(-1, 3)
            all_points_c = torch.cat((all_points_c, local_centroid[0].unsqueeze(0)), dim=0)

            shape = img11_.shape[:2][::-1]

            # vertex
            p_2d1 = (intrinsic @ all_points_c.T).T.cpu().numpy()
            p_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
            p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
            p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
            p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
            p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
            # sample points
            s_2d1 = (intrinsic @ sample_points_c.T).T.cpu().numpy()
            s_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(sample_points_c).T).T.cpu().numpy()
            s_2d1 = s_2d1[:, :2] / s_2d1[:, 2:3]
            s_2d2 = s_2d2[:, :2] / s_2d2[:, 2:3]
            s_2d1 = np.around(s_2d1 * shape).astype(np.int64)
            s_2d2 = np.around(s_2d2 * shape).astype(np.int64)

            vertex_point_color = (0, 0, 255)
            point_thickness = 3

            tri_c = 0
            tri_point_num = torch.cumsum(num_sample_points, dim=0)
            for i_point in range(s_2d1.shape[0]):
                if i_point >= tri_point_num[tri_c]:
                    tri_c += 1
                sample_point_color = tri_colors[tri_c]
                cv2.circle(img11_, s_2d1[i_point], 1, sample_point_color, 1)
                cv2.circle(img21_, s_2d2[i_point], 1, sample_point_color, 1)

            for i_point in range(p_2d1.shape[0]):
                cv2.circle(img11_, p_2d1[i_point], 1, vertex_point_color, point_thickness)
                cv2.circle(img21_, p_2d2[i_point], 1, vertex_point_color, point_thickness)

            return img11_, img21_

        for i in range(len(merged_patch_id_list)):
            img11, img21 = draw_sample(merged_patch_id_list[i], merged_abcd[i].unsqueeze(0), img11, img21)

        if text:
            cv2.putText(img21, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(v_log_root, "merged_patch_{}_iter_{}.jpg").
                    format(merged_patch_id_list, cur_iter_),
                    np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                        np.concatenate([img12, img22], axis=1),
                        ], axis=0)
                    )

        return

    def vis_all_sample(samples_abcd_, merged_patch_id_list, ):
        for i in range(samples_abcd_.shape[0]):
            vis_megred_patch(merged_patch_id_list, samples_abcd_[i], i)

    # eval plane parameter of a patch
    def eval_parameter(v_patch_id, local_plane_parameter, v_planes_, shared_edge=None):
        # 2. Get some variables
        num_plane_sample_ = local_plane_parameter.shape[0]
        num_vertex = len(patch_vertexes_id[v_patch_id])

        local_vertex_pos = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                             v_rays_c[patch_vertexes_id[v_patch_id]])  # 100 * n * 3
        # 3. Construct the triangles
        edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
        local_edge_pos = local_vertex_pos[:, edges_idx]
        local_centroid = intersection_of_ray_and_all_plane(local_plane_parameter,
                                                           v_centroid_rays_c[v_patch_id:v_patch_id + 1])[:, 0]
        triangles_pos = torch.cat((local_edge_pos, local_centroid[:, None, None, :].tile(1, num_vertex, 1, 1)), dim=2)
        triangles_pos = triangles_pos.view(-1, 3, 3)  # (100*num_tri,3,3)

        # 4. Sample points in this plane
        num_sample_points, sample_points_on_face = sample_triangles(100, triangles_pos[:, 0, :],
                                                                    triangles_pos[:, 1, :],
                                                                    triangles_pos[:, 2, :],
                                                                    v_sample_edge=False
                                                                    )
        triangle_normal = normalize_tensor(torch.cross(triangles_pos[:, 0, :] - triangles_pos[:, 1, :],
                                                       triangles_pos[:, 1, :] - triangles_pos[:, 2, :]))
        triangle_normal = triangle_normal.repeat_interleave(num_sample_points, dim=0)
        points_to_tri = torch.arange(num_sample_points.shape[0]).to(device).repeat_interleave(num_sample_points)

        # Collision
        # Viz
        test_collision = False
        if test_collision and cur_iter[v_patch_id] > 20 and v_patch_id == 8:
            collision_checker.save_ply(os.path.join(v_log_root, "collision_test.ply"))
            sample_points_on_face_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(sample_points_on_face).T).T[:, :3]
            collision_flag = collision_checker.check_ray(torch.zeros_like(sample_points_on_face_c2),
                                                         sample_points_on_face_c2)

            remain_flag = ~collision_flag
            sample_points_on_face = sample_points_on_face[remain_flag]
            for i, tri in enumerate(triangles_pos):
                num_sample_points = tri.sum()
            triangle_normal = triangle_normal[remain_flag]
            points_to_tri = torch.arange(num_sample_points.shape[0]).to(device).repeat_interleave(num_sample_points)

            # id_end = num_sample_points[:num_vertex].sum()
            # local_flag = collision_flag[:id_end].cpu().numpy()
            # p2 = (transformation[img_src_id] @ to_homogeneous_tensor(
            #     sample_points_on_face[:id_end]).T).T[:, :3]
            # p2 = p2[:, :2] / p2[:, 2:3]
            # p2 = torch.round(p2 * 800).cpu().numpy().astype(np.int64)
            # viz_img = cv2.cvtColor(v_img2.cpu().numpy(), cv2.COLOR_GRAY2BGR)
            # viz_img[p2[local_flag, 1], p2[local_flag, 0]] = (0, 0, 255)
            # viz_img[p2[~local_flag, 1], p2[~local_flag, 0]] = (0, 255, 255)
            # cv2.imshow("1", viz_img)
            # cv2.waitKey()

        # 5. Compute loss
        # 1) NCC loss
        points_ncc = ncc_loss_computer.compute(sample_points_on_face, triangle_normal, intrinsic,
                                               transformation[img_src_id], v_img1, v_img2)
        triangles_ncc = scatter_mean(points_ncc, points_to_tri, dim=0)
        triangles_ncc = torch.cat((triangles_ncc,
                                   torch.zeros(num_sample_points.shape[0] - triangles_ncc.shape[0]).to(device)))
        triangles_ncc = triangles_ncc.view(num_plane_sample_, -1)
        triangle_weights = num_sample_points.view(num_plane_sample_, -1)
        triangle_weights = triangle_weights / (triangle_weights.sum(dim=-1, keepdim=True) + 1e-6)
        ncc_loss = (triangles_ncc * triangle_weights).mean(dim=-1)

        # filter the sample plane that all its triangles' sample points are collided
        valid_sample = num_sample_points.view(num_plane_sample_, -1).sum(dim=-1) > 0
        ncc_loss[~valid_sample] = ncc_loss[valid_sample].mean()

        # 2) Edge loss
        if shared_edge is not None:
            edges = [[patch_vertexes_id[v_patch_id][i],
                      patch_vertexes_id[v_patch_id][(i + 1) % num_vertex]] for i in range(num_vertex)]
            try:
                shared_edge_idx = edges.index(shared_edge)
            except:
                shared_edge_idx = edges.index(shared_edge[::-1])
            local_edge_pos = torch.cat((local_edge_pos[:, :shared_edge_idx],
                                        local_edge_pos[:, shared_edge_idx + 1:]), dim=1)
        edge_loss, edge_loss_mask, num_samples_per_edge = edge_loss_computer.compute(
                local_edge_pos.reshape(-1, 2, 3),
                intrinsic,
                transformation[img_src_id],
                dilated_gradients1,
                dilated_gradients2,
                v_num_hypothesis=local_edge_pos.shape[0])
        edge_loss[~edge_loss_mask] = torch.inf
        edge_loss = edge_loss.view(num_plane_sample_, -1)
        edge_loss = torch.mean(edge_loss, dim=1)

        # 3) Regularization loss
        reg_loss = compute_regularization(triangles_pos.view(num_plane_sample_, -1, 3, 3), intrinsic,
                                          transformation[img_src_id])

        # 4) Glue loss
        glue_loss = glue_loss_computer.compute(v_patch_id, v_planes_, local_vertex_pos)

        # 5) Regularization loss
        regularization_loss = regularization_loss_computer.compute(v_patch_id, v_planes_)

        # Final loss
        final_loss = ncc_loss * ncc_loss_weight \
                     + edge_loss * edge_loss_weight \
                     + reg_loss * reg_loss_weight \
                     + glue_loss * glue_loss_weight \
                     + regularization_loss * regularization_loss_weight

        return final_loss

    def merge_muti_plane(merged_patch_id_list: list, dual_graph_, patch_vertexes_id_, v_planes_, phase=1):
        # adj_edge = dual_graph_[src_patch_id][adj_patch_id]['adjacent_vertices']
        # merged_patch = init_merged_patch(patch_vertexes_id_[src_patch_id],
        #                                  patch_vertexes_id_[adj_patch_id],
        #                                  adj_edge)

        merged_patch_abcd = v_planes_[merged_patch_id_list].mean(dim=0)

        # eval src parameter of the two patch to be merged
        sum_loss = 0
        for patch_id in merged_patch_id_list:
            sum_loss += eval_parameter(patch_id, v_planes_[patch_id].unsqueeze(0), v_planes_)

        # vis_megred_patch(src_patch_id, adj_patch_id, v_planes[adj_patch_id],-1)

        # optimize the merged patch
        local_iter = 0
        MAX_TOLERENCE = 500
        MAX_ITER = 1000
        num_tolerence = MAX_TOLERENCE

        last_best_loss = None
        history_best = None
        best_abcd = None

        while True:
            # sample new planes
            if phase == 1:
                samples_depth, samples_angle = sample_new_planes(merged_patch_abcd.unsqueeze(0),
                                                                 v_centroid_rays_c[merged_patch_id_list[0]].unsqueeze(
                                                                         0),
                                                                 v_random_g=sample_g)
                samples_abc = angles_to_vectors(samples_angle)  # 1 * 100 * 3
                _, samples_abcd = compute_plane_abcd(v_centroid_rays_c[merged_patch_id_list[0]].unsqueeze(0),
                                                     samples_depth,
                                                     samples_abc)
                samples_abcd = samples_abcd.squeeze(0)
            else:
                pass
                # samples_abcd = sample_new_planes2(merged_patch_abcd.unsqueeze(0),
                #                                   v_centroid_rays_c[merged_patch_id_list[0]].unsqueeze(0),
                #                                   v_random_g=sample_g)
                # samples_abcd = samples_abcd.squeeze(0)
            # vis_all_sample(samples_abcd)

            merged_patch_loss = torch.zeros(1, 100).to(device)
            for patch_id in merged_patch_id_list:
                merged_patch_loss += eval_parameter(patch_id, samples_abcd, v_planes_).view(1, -1)
            merged_patch_loss = merged_patch_loss.squeeze()

            id_best = torch.argmin(merged_patch_loss, dim=-1)

            if id_best != 0:
                merged_patch_abcd = samples_abcd[id_best]

            for patch_id in merged_patch_id_list:
                v_planes_[patch_id] = merged_patch_abcd

            best_loss_ = merged_patch_loss[id_best]

            if last_best_loss is not None:
                delta = best_loss_ - last_best_loss

                if abs(delta) < 1e-3:
                    num_tolerence -= 1
                else:
                    num_tolerence = MAX_TOLERENCE
                    last_best_loss = best_loss_

                if num_tolerence <= 0:
                    if best_loss_ < sum_loss + 0.03:
                        print("merge success")
                        return True, merged_patch_abcd
                    else:
                        print("merge failed")
                        return False, None
            else:
                last_best_loss = merged_patch_loss[id_best]

            if local_iter >= MAX_ITER:
                print("merge failed")
                return False, None

            if local_iter % 100 == 0:
                vis_megred_patch(merged_patch_id_list,
                                 merged_patch_abcd.tile(len(merged_patch_id_list), 1),
                                 cur_iter_=local_iter,
                                 text=str(best_loss_.cpu()) + str(sum_loss.cpu()))
                print(local_iter, id_best, best_loss_)

            local_iter += 1

    def bfs_merge_planes(dual_graph_, patch_vertexes_id_, v_planes_, init_start_patch_id=2):
        # graph_adj = dual_graph_.copy()
        # for node in graph_adj.nodes:
        #     graph_adj.nodes[node].clear()
        #     graph_adj.nodes[node]['merged_plane'] = [node]
        #
        # for u, v in graph_adj.edges:
        #     graph_adj[u][v].clear()
        #     # graph_adj[u][v]['adjacent_vertices'] = [graph_adj[u][v]['adjacent_vertices']]

        # vis src
        vis_megred_patch(list(dual_graph_.nodes), v_planes_, cur_iter_=-1)

        visited = [False] * len(dual_graph_.nodes)

        new_graph = [[] for _ in range(15)]

        ordered_patch_id_list = list(dual_graph_.nodes)
        ordered_patch_id_list = ordered_patch_id_list[init_start_patch_id:-1] + ordered_patch_id_list[
                                                                                0:init_start_patch_id]
        for start_patch_id in ordered_patch_id_list:
            if visited[start_patch_id]:
                continue

            visited[start_patch_id] = True
            new_graph[start_patch_id].append(start_patch_id)
            neighbour_plane_list = list(dual_graph_[start_patch_id].keys())
            queue = deque(neighbour_plane_list)

            while queue:
                neighbour_plane = queue.popleft()
                if visited[neighbour_plane]:
                    continue

                # check merge
                print("Trying to merge {} to {}".format(neighbour_plane, new_graph[start_patch_id]))
                is_merge_success, merged_patch_abcd = \
                    merge_muti_plane(new_graph[start_patch_id] + [neighbour_plane],
                                     dual_graph, patch_vertexes_id, v_planes_.clone())
                # is_merge_success = random.random() > 0.5
                # merged_patch_abcd = torch.zeros(1,4)

                if is_merge_success:
                    visited[neighbour_plane] = True
                    new_graph[start_patch_id].append(neighbour_plane)
                    for p in new_graph[start_patch_id]:
                        v_planes_[p] = merged_patch_abcd

                    # neighbour of explored neighbour
                    for n in list(dual_graph_[neighbour_plane].keys()):
                        if not visited[n]:
                            queue.append(n)
                print(new_graph)

        return new_graph, v_planes_

    # 1. try to optimize the merged plane parameters, then they are assembled naturally
    if not os.path.exists("output/optimized_abcd_list_merged.pkl"):
        merged_graph, optimized_abcd_list = bfs_merge_planes(dual_graph, patch_vertexes_id, v_planes.clone(), 2)
        pickle.dump(optimized_abcd_list.cpu().numpy(), open("output/optimized_abcd_list_merged.pkl", "wb"))
        pickle.dump(merged_graph, open("output/merged_graph.pkl", "wb"))
        save_plane(optimized_abcd_list, v_rays_c, patch_vertexes_id, os.path.join(v_log_root, "merged.ply"))
    else:
        optimized_abcd_list = pickle.load(open("output/optimized_abcd_list_merged.pkl", "rb"))
        merged_graph = pickle.load(open("output/merged_graph.pkl", "rb"))
        optimized_abcd_list = torch.from_numpy(optimized_abcd_list).to(device)

    # get the vertexes id list of each new merged patches
    def merge_patches_vertexes(merged_graph_, patch_vertexes_id_, dual_graph_):
        def merge_one_patch(merged_patch_id_list_):
            shared_edges = set()
            for i in range(len(merged_patch_id_list_)):
                for j in range(i + 1, len(merged_patch_id_list_)):
                    if dual_graph_.has_edge(merged_patch_id_list_[i], merged_patch_id_list_[j]):
                        shared_edge = dual_graph_[merged_patch_id_list_[i]][merged_patch_id_list_[j]][
                            'adjacent_vertices']
                        shared_edges.add(tuple(sorted(shared_edge)))

            merged_polygon_edges = []
            for i in range(len(merged_patch_id_list_)):
                patch_vertexes_id_c = patch_vertexes_id_[merged_patch_id_list_[i]]
                for j in range(len(patch_vertexes_id_c)):
                    edge = [patch_vertexes_id_c[j], patch_vertexes_id_c[(j + 1) % len(patch_vertexes_id_c)]]
                    if tuple(sorted(edge)) not in shared_edges:
                        merged_polygon_edges.append(edge)

            adj_list = [[] for _ in range(max(max(e) for e in merged_polygon_edges) + 1)]
            for x, y in merged_polygon_edges:
                adj_list[x].append(y)
                adj_list[y].append(x)

            visited = set()
            merged_polygon_vertexes_id = []

            def dfs(u):
                visited.add(u)
                merged_polygon_vertexes_id.append(u)
                for v in adj_list[u]:
                    if v not in visited:
                        dfs(v)

            dfs(merged_polygon_edges[0][0])

            # start_edge = merged_polygon_edges.pop(0)
            # merged_polygon_vertexes_id.extend(start_edge)
            # while merged_polygon_edges:
            #     for i, edge in enumerate(merged_polygon_edges):
            #         if edge[0] == merged_polygon_vertexes_id[-1]:
            #             merged_polygon_vertexes_id.append(edge[1])
            #             merged_polygon_edges.pop(i)
            #             break
            #         elif edge[1] == merged_polygon_vertexes_id[-1]:
            #             merged_polygon_vertexes_id.append(edge[0])
            #             merged_polygon_edges.pop(i)
            #             break
            # assert (merged_polygon_vertexes_id[0] == merged_polygon_vertexes_id[-1])
            # merged_polygon_vertexes_id.pop(-1)
            return merged_polygon_vertexes_id

        merged_patch_vertexes_id_ = []
        for merged_patch_id_list in merged_graph_:
            if len(merged_patch_id_list) == 0:
                continue
            elif len(merged_patch_id_list) == 1:
                merged_patch_vertexes_id_.append(patch_vertexes_id_[merged_patch_id_list[0]])
            else:
                merged_patch_vertexes_id_.append(merge_one_patch(merged_patch_id_list))
        return merged_patch_vertexes_id_

    # merge the planes, get the new graph
    merged_patch_vertexes_id = merge_patches_vertexes(merged_graph, patch_vertexes_id, dual_graph)
    valid_idx = [i for i in range(len(merged_graph)) if len(merged_graph[i]) != 0]
    merged_graph = [merged_graph[i] for i in valid_idx]
    optimized_abcd_list = optimized_abcd_list[valid_idx]

    # create the new graph
    def create_graph(merged_graph_, merged_patch_vertexes_id_, dual_graph_):
        graph = nx.Graph()

        # 以每个多边形为节点，节点属性是多边形的顶点列表
        for i, merged_patch in enumerate(merged_patch_vertexes_id_):
            graph.add_node(i, id_vertex=merged_patch,
                           id_in_original_array=merged_graph_[i],
                           face_center=dual_graph_.nodes[merged_graph_[i][0]]['face_center'],
                           ray_c=dual_graph_.nodes[merged_graph_[i][0]]['ray_c'])

        # 查找相邻多边形并创建边，边的属性是它们共享的节点
        for i, merged_patch1 in enumerate(merged_patch_vertexes_id_):
            for j, merged_patch2 in enumerate(merged_patch_vertexes_id_[i + 1:], i + 1):
                adjacent_vertices = list(set(merged_patch1) & set(merged_patch2))
                if len(adjacent_vertices) > 0:
                    graph.add_edge(i, j, adjacent_vertices=adjacent_vertices)
        return graph

    merged_dual_graph = create_graph(merged_graph, merged_patch_vertexes_id, dual_graph)

    # 2. assemble the unmerged planes
    # 修改eval_parameter，有两种情况，一种是合并平面，一种是组装平面
    # 组装平面需要sample两个平面的法向，然后sample两个点深度
    def bfs_assemble_planes(dual_graph_, patch_vertexes_id_, v_planes_, new_graph_):
        new_graph_ = [item for item in new_graph_ if len(item) > 0]
        # merged_graph[1], merged_graph[3]
        pass

    bfs_assemble_planes(dual_graph, patch_vertexes_id, v_planes.clone(), merged_graph)
    return merged_dual_graph, optimized_abcd_list
