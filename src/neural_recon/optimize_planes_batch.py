import os

import cv2
import numpy as np
import torch
from torch_scatter import scatter_mean

from shared.common_utils import normalize_tensor, to_homogeneous_tensor
from src.neural_recon.geometric_util import angles_to_vectors, compute_plane_abcd, intersection_of_ray_and_all_plane, \
    intersection_of_ray_and_plane
from src.neural_recon.io_utils import generate_random_color, save_plane
from src.neural_recon.loss_utils import compute_regularization, Glue_loss_computer, \
    Bilateral_ncc_computer, Edge_loss_computer
from src.neural_recon.collision_checker import Collision_checker
from src.neural_recon.sample_utils import sample_new_planes, sample_triangles


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
    reg_loss_weight = 10
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
    return optimized_abcd_list


def local_assemble(v_planes, v_rays_c, v_centroid_rays_c, dual_graph,
                   imgs, dilated_gradients1, dilated_gradients2,
                   transformation,
                   intrinsic,
                   v_c1_2_c2,
                   v_log_root
                   ):
    # 1. Prepare variables

    # 2. Choose initial patch

    while True:
        # 3. Select nearby patch and expand

        # 4. Global optimize
        pass

    return
