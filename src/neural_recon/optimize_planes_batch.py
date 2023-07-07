import os
import random

import cv2
import networkx as nx
import numpy as np
import torch
import pickle
from torch_scatter import scatter_mean

from shared.common_utils import normalize_tensor, to_homogeneous_tensor
from src.neural_recon.geometric_util import angles_to_vectors, compute_plane_abcd, intersection_of_ray_and_all_plane, \
    intersection_of_ray_and_plane
from src.neural_recon.io_utils import generate_random_color, save_plane
from src.neural_recon.loss_utils import compute_regularization, Glue_loss_computer, \
    Bilateral_ncc_computer, Edge_loss_computer
from src.neural_recon.collision_checker import Collision_checker
from src.neural_recon.sample_utils import sample_new_planes, sample_triangles
from src.neural_recon.geometric_util import fit_plane_svd
from collections import deque


def optimize_planes_batch(initialized_planes, v_rays_c, v_centroid_rays_c, dual_graph,
                          imgs, dilated_gradients1, dilated_gradients2,
                          transformation,
                          intrinsic,
                          v_c1_2_c2,
                          v_log_root
                          ):
    # Prepare data
    device = initialized_planes.device
    patch_num = len(initialized_planes)
    patches_list = dual_graph.nodes  # each patch = id_vertexes
    patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]
    optimized_abcd_list = initialized_planes.clone()

    cur_iter = [0] * patch_num
    best_loss = [None] * patch_num
    delta = [None] * patch_num
    MAX_TOLERENCE = 500
    num_tolerence = [MAX_TOLERENCE] * patch_num

    num_plane_sample = 100
    img_src_id = 0

    v_img1 = imgs[0]
    v_img2 = imgs[img_src_id + 1]

    tri_colors = [generate_random_color() for _ in range(100)]
    sample_g = torch.Generator(device)
    sample_g.manual_seed(0)
    end_flag = [0] * patch_num

    ncc_loss_weight = 1
    edge_loss_weight = 10
    reg_loss_weight = 1
    glue_loss_weight = 1
    edge_loss_computer = Edge_loss_computer(v_sample_density=0.001)
    glue_loss_computer = Glue_loss_computer(dual_graph, v_rays_c)
    ncc_loss_computer = Bilateral_ncc_computer(
        v_enable_spatial_weights=True,
        v_enable_color_weights=True,
        v_window_size=7
    )
    collision_checker = Collision_checker()

    while True:
        # 1. Sample new hypothesis from 1) propagation 2) random perturbation
        samples_depth, samples_angle = sample_new_planes(optimized_abcd_list,
                                                         v_centroid_rays_c,
                                                         dual_graph,
                                                         sample_g)

        # Right
        # samples_depth[6, 0] = 4.423
        # samples_angle[6, 0, 0] = 3.2087
        # samples_angle[6, 0, 1] = 3.0404

        # Middle
        # samples_depth[11, 0] = 4.35
        # samples_angle[11, 0, 0] = 2.0643
        # samples_angle[11, 0, 1] = 0.4344
        samples_abc = angles_to_vectors(samples_angle)  # n * 100 * 3
        samples_intersection, samples_abcd = compute_plane_abcd(v_centroid_rays_c, samples_depth, samples_abc)

        if sum(end_flag) == patch_num:
            break

        # Start to optimize
        for v_patch_id in range(patch_num):
            if end_flag[v_patch_id] == 1:
                continue
            # if v_patch_id != 1 and v_patch_id != 8:
            # if v_patch_id != 11:
            #     continue
            # if v_patch_id != 1:
            #     continue

            # 2. Get some variables
            num_vertex = len(patch_vertexes_id[v_patch_id])
            local_plane_parameter = samples_abcd[v_patch_id]
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

                id_end = num_sample_points[:num_vertex].sum()
                local_flag = collision_flag[:id_end].cpu().numpy()
                p2 = (transformation[0] @ to_homogeneous_tensor(
                    sample_points_on_face[:id_end]).T).T[:, :3]
                p2 = p2[:, :2] / p2[:, 2:3]
                p2 = torch.round(p2 * 800).cpu().numpy().astype(np.int64)
                viz_img = cv2.cvtColor(v_img2.cpu().numpy(), cv2.COLOR_GRAY2BGR)
                viz_img[p2[local_flag, 1], p2[local_flag, 0]] = (0, 0, 255)
                viz_img[p2[~local_flag, 1], p2[~local_flag, 0]] = (0, 255, 255)
                cv2.imshow("1", viz_img)
                cv2.waitKey()

            # 5. Compute loss
            # 1) NCC loss
            points_ncc = ncc_loss_computer.compute(sample_points_on_face, triangle_normal, intrinsic,
                                                   transformation[img_src_id], v_img1, v_img2)
            triangles_ncc = scatter_mean(points_ncc, points_to_tri, dim=0)
            triangles_ncc = triangles_ncc.view(num_plane_sample, -1)
            triangle_weights = num_sample_points.view(num_plane_sample, -1)
            triangle_weights = triangle_weights / triangle_weights.sum(dim=-1, keepdim=True)
            ncc_loss = (triangles_ncc * triangle_weights).mean(dim=-1)

            # 2) Edge loss
            edge_loss, edge_loss_mask, num_samples_per_edge = edge_loss_computer.compute(
                local_edge_pos.reshape(-1, 2, 3),
                intrinsic,
                transformation[img_src_id],
                dilated_gradients1,
                dilated_gradients2,
                v_num_hypothesis=local_edge_pos.shape[0])
            edge_loss[~edge_loss_mask] = torch.inf
            edge_loss = edge_loss.view(num_plane_sample, -1)
            edge_loss = torch.mean(edge_loss, dim=1)

            # 3) Regularization loss
            reg_loss = compute_regularization(triangles_pos.view(num_plane_sample, -1, 3, 3), intrinsic,
                                              transformation[img_src_id])

            # 4) Glue loss
            glue_loss = glue_loss_computer.compute(v_patch_id, optimized_abcd_list, local_vertex_pos)

            # Final loss
            final_loss = ncc_loss * ncc_loss_weight \
                         + edge_loss * edge_loss_weight \
                         + reg_loss * reg_loss_weight \
                         + glue_loss * glue_loss_weight
            # final_loss = weighted_triangle_loss

            # 6. Select the best and update `optimized_abcd_list`
            id_best = torch.argmin(final_loss, dim=-1)
            optimized_abcd_list[v_patch_id] = local_plane_parameter[id_best]

            # 7. Visualize
            if True:
                num_sample_points_per_triangle = num_sample_points.view(num_plane_sample, -1).sum(dim=-1)
                num_sample_points_per_triangle = torch.cumsum(num_sample_points_per_triangle, dim=0)
                if id_best == 0:
                    sample_points_c = sample_points_on_face[0:num_sample_points_per_triangle[id_best]]
                else:
                    sample_points_c = sample_points_on_face[
                                      num_sample_points_per_triangle[id_best - 1]:num_sample_points_per_triangle[
                                          id_best]]
                all_points_c = local_vertex_pos[id_best].view(-1, 3)
                all_points_c = torch.cat((all_points_c, local_centroid[id_best].unsqueeze(0)), dim=0)

                def vis_(all_points_c_, sample_points_c_, patch_id_, cur_iter_, tri_point_num, text=None):
                    ref_img = imgs[0].cpu().numpy() * 255
                    src_imgs = imgs[1:].cpu().numpy() * 255
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                    img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                    shape = img11.shape[:2][::-1]
                    # vertex
                    p_2d1 = (intrinsic @ all_points_c_.T).T.cpu().numpy()
                    p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c_).T).T.cpu().numpy()
                    p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                    p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
                    p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                    p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                    # sample points
                    s_2d1 = (intrinsic @ sample_points_c_.T).T.cpu().numpy()
                    s_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_c_).T).T.cpu().numpy()
                    s_2d1 = s_2d1[:, :2] / s_2d1[:, 2:3]
                    s_2d2 = s_2d2[:, :2] / s_2d2[:, 2:3]
                    s_2d1 = np.around(s_2d1 * shape).astype(np.int64)
                    s_2d2 = np.around(s_2d2 * shape).astype(np.int64)

                    line_color = (0, 0, 255)
                    line_thickness = 2
                    vertex_point_color = (0, 0, 255)
                    sample_point_color = (0, 255, 0)
                    point_thickness = 3

                    tri_c = 0
                    tri_point_num = torch.cumsum(tri_point_num, dim=0)
                    for i_point in range(s_2d1.shape[0]):
                        if i_point >= tri_point_num[tri_c]:
                            tri_c += 1
                        sample_point_color = tri_colors[tri_c]
                        cv2.circle(img11, s_2d1[i_point], 1, sample_point_color, 1)
                        cv2.circle(img21, s_2d2[i_point], 1, sample_point_color, 1)

                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point], 1, vertex_point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point], 1, vertex_point_color, point_thickness)

                    if text:
                        cv2.putText(img21, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(os.path.join(v_log_root, "patch_{}_iter_{}.jpg").format(patch_id_, cur_iter_),
                                np.concatenate([
                                    np.concatenate([img11, img21], axis=1),
                                    np.concatenate([img12, img22], axis=1),
                                ], axis=0)
                                )

                # save the init result
                if cur_iter[v_patch_id] == 0:
                    print("Patch {:3d} cur_iter{:4d}:{:.4f};".format(v_patch_id,
                                                                     cur_iter[v_patch_id],
                                                                     final_loss[0].cpu().item()))
                    all_points_c_init = local_vertex_pos[0].view(-1, 3)
                    sample_points_c_init = sample_points_on_face[0:num_sample_points_per_triangle[0]]
                    vis_(all_points_c_init, sample_points_c_init, v_patch_id, cur_iter[v_patch_id],
                         num_sample_points.view(100, -1)[0], str(final_loss[0].cpu()))

                if cur_iter[v_patch_id] % 50 == 0:
                    vis_(all_points_c, sample_points_c, v_patch_id, cur_iter[v_patch_id] + 1,
                         num_sample_points.view(100, -1)[id_best], str(final_loss[id_best].cpu()))

            # 8. Control the end of the optimization
            cur_loss = final_loss[id_best].cpu().item()
            if best_loss[v_patch_id] is not None:
                delta[v_patch_id] = cur_loss - best_loss[v_patch_id]
                print("Patch/Iter: {:3d}, {:4d}; Loss: {:.4f} Best: {:.4f} Delta: {:.4f}".format(
                    v_patch_id,
                    cur_iter[v_patch_id] + 1,
                    cur_loss,
                    best_loss[v_patch_id],
                    delta[v_patch_id])
                )
                if delta[v_patch_id] < 1e-4:
                    num_tolerence[v_patch_id] -= 1
                else:
                    num_tolerence[v_patch_id] = MAX_TOLERENCE
                    best_loss[v_patch_id] = cur_loss

                if num_tolerence[v_patch_id] <= 0:
                    end_flag[v_patch_id] = 1
            else:
                best_loss[v_patch_id] = cur_loss

            cur_iter[v_patch_id] += 1

        # Update triangles
        final_triangles_list = []
        for v_patch_id in range(patch_num):
            local_vertex_pos = intersection_of_ray_and_all_plane(optimized_abcd_list[v_patch_id:v_patch_id + 1],
                                                                 v_rays_c[patch_vertexes_id[v_patch_id]])
            num_vertex = len(patch_vertexes_id[v_patch_id])
            edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
            local_edge_pos = local_vertex_pos[:, edges_idx][0]
            local_centroid = intersection_of_ray_and_plane(
                optimized_abcd_list[v_patch_id:v_patch_id + 1],
                v_centroid_rays_c[v_patch_id:v_patch_id + 1])[1]
            triangles_pos = torch.cat((
                local_edge_pos, local_centroid[:, None, :].tile(num_vertex, 1, 1)), dim=1)
            final_triangles_list.append(triangles_pos)
        final_triangles = torch.cat(final_triangles_list, dim=0)
        final_triangles_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
            final_triangles).transpose(1, 2)).transpose(1, 2)[:, :, :3]
        collision_checker.clear()
        collision_checker.add_triangles(final_triangles_c2)
        continue

    save_plane(optimized_abcd_list, v_rays_c, patch_vertexes_id,
               file_path=os.path.join(v_log_root, "optimized.ply"))

    # save optimized_abcd_list
    with open("output/optimized_abcd_list.pkl", "wb") as f:
        pickle.dump(optimized_abcd_list, f)

    return optimized_abcd_list


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
    img_src_id = 0

    v_img1 = imgs[0]
    v_img2 = imgs[img_src_id + 1]

    tri_colors = [generate_random_color() for _ in range(100)]
    sample_g = torch.Generator(device)

    ncc_loss_weight = 1
    edge_loss_weight = 10
    reg_loss_weight = 1
    glue_loss_weight = 1
    edge_loss_computer = Edge_loss_computer(v_sample_density=0.001)
    glue_loss_computer = Glue_loss_computer(dual_graph, v_rays_c)
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
            p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
            p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
            p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
            p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
            p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
            # sample points
            s_2d1 = (intrinsic @ sample_points_c.T).T.cpu().numpy()
            s_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_c).T).T.cpu().numpy()
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

    # eval src plane parameter of a patch
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
        triangles_pos = torch.cat((local_edge_pos, local_centroid[:, None, None, :].tile(1, num_vertex, 1, 1)),
                                  dim=2)
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

            id_end = num_sample_points[:num_vertex].sum()
            local_flag = collision_flag[:id_end].cpu().numpy()
            p2 = (transformation[0] @ to_homogeneous_tensor(
                sample_points_on_face[:id_end]).T).T[:, :3]
            p2 = p2[:, :2] / p2[:, 2:3]
            p2 = torch.round(p2 * 800).cpu().numpy().astype(np.int64)
            viz_img = cv2.cvtColor(v_img2.cpu().numpy(), cv2.COLOR_GRAY2BGR)
            viz_img[p2[local_flag, 1], p2[local_flag, 0]] = (0, 0, 255)
            viz_img[p2[~local_flag, 1], p2[~local_flag, 0]] = (0, 255, 255)
            cv2.imshow("1", viz_img)
            cv2.waitKey()

        # 5. Compute loss
        # 1) NCC loss
        points_ncc = ncc_loss_computer.compute(sample_points_on_face, triangle_normal, intrinsic,
                                               transformation[img_src_id], v_img1, v_img2)
        triangles_ncc = scatter_mean(points_ncc, points_to_tri, dim=0)
        triangles_ncc = triangles_ncc.view(num_plane_sample_, -1)
        triangle_weights = num_sample_points.view(num_plane_sample_, -1)
        triangle_weights = triangle_weights / triangle_weights.sum(dim=-1, keepdim=True)
        ncc_loss = (triangles_ncc * triangle_weights).mean(dim=-1)

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

        # Final loss
        final_loss = ncc_loss * ncc_loss_weight \
                     + edge_loss * edge_loss_weight \
                     + reg_loss * reg_loss_weight \
                     + glue_loss * glue_loss_weight

        return final_loss

    def merge_muti_plane(merged_patch_id_list: list, dual_graph_, patch_vertexes_id_, v_planes_):
        # adj_edge = dual_graph_[src_patch_id][adj_patch_id]['adjacent_vertices']
        # merged_patch = init_merged_patch(patch_vertexes_id_[src_patch_id],
        #                                  patch_vertexes_id_[adj_patch_id],
        #                                  adj_edge)

        merged_patch_abcd = v_planes_[merged_patch_id_list].mean(dim=0)

        # eval src parameter of the two patch to be merged
        # for i in range(10000):
        #     patch_id = 0
        #     print(eval_parameter(patch_id, v_planes[patch_id].unsqueeze(0), v_planes_).item())

        sum_loss = 0
        for patch_id in merged_patch_id_list:
            sum_loss += eval_parameter(patch_id, v_planes_[patch_id].unsqueeze(0), v_planes_)

        # vis_megred_patch(src_patch_id, adj_patch_id, v_planes[adj_patch_id],-1)

        # optimize the merged patch
        local_iter = 0
        MAX_TOLERENCE = 300
        num_tolerence = MAX_TOLERENCE

        last_best_loss = None
        history_best = None
        best_abcd = None

        while True:
            # sample new planes
            samples_depth, samples_angle = sample_new_planes(merged_patch_abcd.unsqueeze(0),
                                                             v_centroid_rays_c[merged_patch_id_list[0]].unsqueeze(0),
                                                             v_random_g=sample_g)
            samples_abc = angles_to_vectors(samples_angle)  # 1 * 100 * 3
            _, samples_abcd = compute_plane_abcd(v_centroid_rays_c[merged_patch_id_list[0]].unsqueeze(0),
                                                 samples_depth,
                                                 samples_abc)
            samples_abcd = samples_abcd.squeeze(0)

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
                delta = last_best_loss - best_loss_
                if delta < 1e-6:
                    num_tolerence -= 1
                else:
                    num_tolerence = MAX_TOLERENCE
                    last_best_loss = best_loss_

                if num_tolerence <= 0:
                    if best_loss_ < sum_loss + 0.02:
                        print("merge success")
                        return True, merged_patch_abcd
                    else:
                        print("merge failed")
                        return False, None
            else:
                last_best_loss = merged_patch_loss[id_best]

            if local_iter % 100 == 0:
                vis_megred_patch(merged_patch_id_list,
                                 merged_patch_abcd.tile(len(merged_patch_id_list),1),
                                 cur_iter_=local_iter,
                                 text=str(best_loss_.cpu())+str(sum_loss.cpu()))
                print(local_iter, id_best, best_loss_)

            if local_iter == 0 and sum_loss == best_loss_:
                print('debug')
            local_iter += 1

    # g = nx.Graph()
    # node
    # - boundary:
    # - id_source_patch:
    import random
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

        merged_plane_list = torch.arange(patch_num)
        merged_record = [-1] * patch_num

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
                if int(neighbour_plane) == 6:
                    print("debug")
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
        #   Construct graph adj
        return new_graph, v_planes_

    #optimized_abcd_list = pickle.load(open("output/optimized_abcd_list_merged.pkl", "rb"))
    #optimized_abcd_list = torch.from_numpy(optimized_abcd_list).to(device)

    merged_graph, optimized_abcd_list = bfs_merge_planes(dual_graph, patch_vertexes_id, v_planes.clone(), 2)

    save_plane(optimized_abcd_list, v_rays_c, patch_vertexes_id,
               file_path=os.path.join(v_log_root, "optimized.ply"))

    return

    # bfs_merge_planes(dual_graph, patch_vertexes_id, v_planes)
    temp, merged_patch_parameter = merge_two_plane(2, 3, dual_graph, patch_vertexes_id, v_planes)
    if temp:
        print("merge success")
    else:
        print("merge failed")

        pass

    return
