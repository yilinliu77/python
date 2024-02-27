from copy import copy

import cv2
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import networkx as nx
from torch_scatter import scatter_add, scatter_min, scatter_mean, scatter_sum

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img, sample_img_prediction2

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility

from src.neural_recon.sample_utils import sample_points_2d, Sample_new_planes

import math
import pickle
import gc

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


def initialize_patches(rays_c, ray_distances_c, v_vertex_id_per_face):
    initialized_vertices = rays_c * ray_distances_c[:, None]

    if True:
        points = initialized_vertices.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud('output/initialized_vertices.ply', pcd)

    # abcd
    plane_parameters = []
    for vertex_id in v_vertex_id_per_face:
        pos_vertexes = initialized_vertices[vertex_id]
        # assert (len(pos_vertexes) >= 3)
        # # a) 3d vertexes of each patch -> fitting plane
        # p_abcd = fit_plane_svd(pos_vertexes)
        abc = torch.mean(pos_vertexes, dim=0)
        d = -torch.dot(abc, abc)
        p_abcd = torch.cat((abc, d.unsqueeze(0)), dim=-1)
        plane_parameters.append(p_abcd)

    return torch.stack(plane_parameters, dim=0)


def determine_valid_edges(v_graph, v_img, v_gradient):
    img_np = v_img.cpu().numpy()
    gradient_np = v_gradient.cpu().numpy()

    for edge in v_graph.edges():
        pos1 = v_graph.nodes[edge[0]]["pos_2d"]
        pos2 = v_graph.nodes[edge[1]]["pos_2d"]
        pos = torch.from_numpy(np.stack((pos1, pos2), axis=0).astype(np.float32)).to(v_img.device).unsqueeze(0)
        pixels_endpoints = sample_img(v_img[None, None, :, :], pos)[0]
        v_graph.nodes[edge[0]]["is_black"] = (pixels_endpoints[0] < 0.05).all()
        v_graph.nodes[edge[1]]["is_black"] = (pixels_endpoints[1] < 0.05).all()

        length = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)

        ns, s = sample_points_2d(pos,
                                 torch.tensor([1000] * pos.shape[0], dtype=torch.long, device=pos.device),
                                 v_img_width=v_img.shape[1], v_vertical_length=1)
        pixels = sample_img(v_img[None, None, :, :], s[None, :])[0]
        gradient = sample_img(v_gradient[None, :].permute(0, 3, 1, 2), s[None, :, :])[0]
        mean_pixels = scatter_mean(pixels,
                                   torch.arange(ns.shape[0], device=pos.device).repeat_interleave(ns),
                                   dim=0)
        mean_gradient = (torch.linalg.norm(gradient, dim=-1)).mean()
        black_rate = (pixels.squeeze(1) == 0).sum() / pixels.squeeze(1).shape[0]
        gradient_free_rate = (gradient.mean(dim=1) == 0).sum() / gradient.mean(dim=1).shape[0]
        # v_graph.edges[edge]["is_black"] = mean_pixels < 0.2
        v_graph.edges[edge]["is_black"] = black_rate > 0.5
        v_graph.edges[edge]["is_gradient_free"] = gradient_free_rate > 0.01
        v_graph.edges[edge]["gradient"] = mean_gradient
        v_graph.edges[edge]["gradient_free_rate"] = gradient_free_rate
        v_graph.edges[edge]["is_short_length"] = length < 0.01

        if False:
            s_pixel = (s * torch.tensor(v_img.shape[::1], device=s.device)).round().long()
            for point in s_pixel:
                x, y = point.tolist()  # 注意：在图像中，y 对应行数，x 对应列数
                if 0 <= y < img_np.shape[0] and 0 <= x < img_np.shape[1]:  # 确保坐标在图像范围内
                    img_np[y, x] = 0  # BGR 格式，将点设为绿色

            cv2.imshow("1", img_np)
            cv2.waitKey()
        pass


# Remove the redundant (1.any vertices connect with image boundary 2.any edge in black) face and edges in the graph
# And build the dual graph in order to navigate between patches
# v_graph.nodes[i]['valid_flag']: 0 if the vertex is in the image boundary
# v_graph.edges[(i, j)]['valid_flag']: 0 if any endpoint of the edge is in the image boundary
# v_graph.graph["face_flags"][i]: 0 if any node of the face is in the image boundary
# v_graph.edges[(i, j)]['is_black']: 1 if the edge is in the black region
def fix_graph(v_graph, ref_img, is_visualize=False):
    dual_graph = nx.Graph()

    id_original_to_current = {}
    for id_face, face in enumerate(v_graph.graph["faces"]):
        # v_graph.graph["face_flags"][id_face] is set to 0 if any vertices in the image boundary
        if not v_graph.graph["face_flags"][id_face]:
            continue
        has_black = False
        for idx, id_start in enumerate(face):
            id_end = face[(idx + 1) % len(face)]
            if v_graph.edges[(id_start, id_end)]["is_black"]:
                has_black = True
                break
        if has_black:
            continue

        id_original_to_current[id_face] = len(dual_graph.nodes)
        dual_graph.add_node(len(dual_graph.nodes),
                            id_vertex=face,
                            sub_faces_id=v_graph.graph["sub_faces_id"][id_face],
                            sub_faces=v_graph.graph["sub_faces"][id_face],
                            id_in_original_array=id_face,
                            face_center=v_graph.graph['face_center'][id_face],
                            ray_c=v_graph.graph['ray_c'][id_face])

    node_without_neighbour = []
    for node in dual_graph.nodes():
        faces = dual_graph.nodes[node]["id_vertex"]
        # for each edge in current face
        num_neighbour = 0
        for idx, id_start in enumerate(faces):
            id_end = faces[(idx + 1) % len(faces)]
            # get the adjacent face of the edge (only one beyond the current idx face)
            t = copy(v_graph.edges[(id_start, id_end)]["id_face"])
            t.remove(dual_graph.nodes[node]["id_in_original_array"])
            # no need to consider the face not in the dual_graph
            if t[0] in id_original_to_current:
                num_neighbour += 1
                # add a edge between current face and adjacent face to dual graph
                edge = (node, id_original_to_current[t[0]])
                # check, merge if the edge is gradient free
                if edge not in dual_graph.edges():
                    id_points_in_another_face = dual_graph.nodes[id_original_to_current[t[0]]]["id_vertex"]
                    adjacent_vertices = []
                    id_cur = idx
                    while True:
                        adjacent_vertices.append(id_start)
                        id_cur += 1
                        if id_cur >= len(faces):
                            adjacent_vertices.append(id_end)
                            break
                        id_start = faces[id_cur]
                        id_end = faces[(id_cur + 1) % len(faces)]
                        if id_end not in id_points_in_another_face:
                            adjacent_vertices.append(id_start)
                            break

                    # adjacent_vertices.append(id_start)
                    dual_graph.add_edge(edge[0], edge[1], adjacent_vertices=adjacent_vertices)
        if num_neighbour == 0:
            node_without_neighbour.append(node)

    # node_without_neighbour： may share only one vertex with other patch
    if dual_graph.number_of_nodes() > 1 and node_without_neighbour:
        for node in node_without_neighbour:
            dual_graph.remove_node(node)
        nx.convert_node_labels_to_integers(dual_graph)

    v_graph.graph["dual_graph"] = dual_graph

    if is_visualize:
        for idx, id_face in enumerate(dual_graph.nodes):
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]
            face = dual_graph.nodes[id_face]["id_vertex"]
            print("{}/{}, points_num: {}, sub_faces: {}".format(idx,
                                                                len(dual_graph.nodes),
                                                                len(face),
                                                                dual_graph.nodes[id_face]["sub_faces_id"]))

            # draw current merged face
            gradient_list = []
            for idx, id_start in enumerate(face):
                id_end = face[(idx + 1) % len(face)]
                pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                print("({}, {}) pos1: {}, pos2: {}, length: {} gradient: {} gradient_free_rate: {}".format(
                        id_start, id_end, pos1, pos2, np.linalg.norm(pos1 - pos2),
                        v_graph.edges[(id_start, id_end)]["gradient"],
                        v_graph.edges[(id_start, id_end)]["gradient_free_rate"]))
                gradient_list.append(v_graph.edges[(id_start, id_end)]["gradient"])
                if v_graph.edges[(id_start, id_end)]["is_gradient_free"]:
                    cv2.line(img11, pos1, pos2, (255, 0, 0), 2)
                else:
                    cv2.line(img11, pos1, pos2, (0, 0, 255), 2)
                cv2.circle(img11, pos1, 2, (0, 0, 255), 2)
                cv2.circle(img11, pos2, 2, (0, 0, 255), 2)

            # draw sub_face of merged face
            for face in dual_graph.nodes[id_face]["sub_faces"]:
                # draw a sub_face
                for idx, id_start in enumerate(face):
                    id_end = face[(idx + 1) % len(face)]
                    pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                    pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                    cv2.line(img11, pos1, pos2, (0, 0, 255), 1)

            # draw neighbor faces
            for id_another_face in dual_graph[id_face]:
                face = dual_graph.nodes[id_another_face]["id_vertex"]
                for idx, id_start in enumerate(face):
                    id_end = face[(idx + 1) % len(face)]
                    pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                    pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                    cv2.line(img11, pos1, pos2, (0, 255, 255), 1)

                for id_vertex in dual_graph[id_face][id_another_face]["adjacent_vertices"]:
                    pos1 = np.around(v_graph.nodes[id_vertex]["pos_2d"] * shape).astype(np.int64)
                    cv2.circle(img11, pos1, 2, (0, 255, 0), 2)

            cv2.circle(img11, np.around(dual_graph.nodes[id_face]['face_center'] * shape).astype(np.int64), 2,
                       (0, 0, 255), 2)

            cv2.imshow("1", img11)
            cv2.waitKey()
    return


class CosineAnnealingScaleFactorScheduler:
    def __init__(self, initial_sf=3.0, initial_sf_mult=0.99, T_0=12, T_mult=1.1, min_sf=0, random_jump=True):
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
        self.random_jump = random_jump

    def get_sf(self):
        if self.cur_iter != 0 and self.cur_iter % self.T_0 == 0:
            self.cur_iter = 0
            self.cur_cycle += 1

            if self.cur_cycle != 0 and self.cur_cycle % 2 == 0:
                self.initial_sf *= self.initial_sf_mult

            # if self.random_jump:
            #     self.init_sf_cur_cycle = self.initial_sf * np.clip(np.random.standard_cauchy(), 0.1, 3.0)
            # else:
            #     self.init_sf_cur_cycle = self.initial_sf

            new_T_0 = int(self.T_0 * self.T_mult)
            self.T_0 = new_T_0 if new_T_0 % 2 == 0 else new_T_0 + 1

        sf = self.min_sf + (self.init_sf_cur_cycle - self.min_sf) * (
                1 + math.cos(math.pi * (self.cur_iter % self.T_0) / self.T_0)) / 2

        # standard flying per iter
        if self.random_jump:
            sf = sf * np.clip(np.random.standard_cauchy(), 0.1, 3.0)

        self.sf_history.append(sf)
        self.cur_iter += 1
        return sf


class GraphPlaneOptimiser:
    def __init__(self, ref_img_id: int, graph: nx.Graph, ref_Image: Image, src_Images: list[Image],
                 v_log_root, device, use_cache=False, use_group=False):
        self.ref_img_id = ref_img_id
        self.graph = graph
        self.device = device
        self.v_log_root = v_log_root

        # constant data
        self.ref_Image = ref_Image
        self.intrinsic = torch.from_numpy(self.ref_Image.intrinsic).to(self.device).to(torch.float32)
        self.extrinsic_ref_cam = torch.from_numpy(self.ref_Image.extrinsic).to(self.device).to(torch.float32)

        self.ref_img = cv2.imread(self.ref_Image.img_path, cv2.IMREAD_GRAYSCALE)
        self.ref_img_tensor = torch.from_numpy(self.ref_img).to(self.device).to(torch.float32)

        gy, gx = torch.gradient(self.ref_img_tensor)
        self.gradients1 = torch.stack((gx, gy), dim=-1)
        if not use_cache:
            determine_valid_edges(graph, self.ref_img_tensor, self.gradients1)
            fix_graph(graph, self.ref_img, is_visualize=False)

        # variable data, we may update the src img id
        self.src_imgs = None
        self.imgs = None
        self.projection2 = None
        self.transformation = None
        self.c1_2_c2_list = None
        self.camera_coor_in_c1 = None
        self.gradients2 = None
        self.prepare_data(src_Images)

        # graph data
        rays_c = [None] * len(graph.nodes)
        ray_distances_c = [None] * len(graph.nodes)
        for idx, id_points in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_points]["ray_c"]
            ray_distances_c[idx] = graph.nodes[id_points]["distance"]
        self.rays_c = torch.from_numpy(np.stack(rays_c)).to(self.device).to(torch.float32)
        self.ray_distances_c = torch.from_numpy(np.stack(ray_distances_c)).to(self.device).to(torch.float32)

        self.dual_graph = graph.graph["dual_graph"]
        self.centroid_rays_c = torch.from_numpy(
                np.stack([self.dual_graph.nodes[id_node]["ray_c"] for id_node in self.dual_graph])).to(
                self.device).to(torch.float32)
        self.src_faces_centroid_rays_c = torch.from_numpy(graph.graph["src_face_ray_c"]).to(self.device).to(
                torch.float32)

        self.patches_list = self.dual_graph.nodes
        self.patch_vertexes_id = [self.patches_list[i]['id_vertex'] for i in range(len(self.patches_list))]

        if use_cache:
            print("Load optimized_abcd_list from cache")
            assert 'optimized_abcd_list' in self.dual_graph.graph
            self.initialized_planes = self.dual_graph.graph['optimized_abcd_list']
            # self.initialized_planes = initialize_patches(self.rays_c,
            #                                              self.ray_distances_c,
            #                                              self.patch_vertexes_id)  # (num_patch, 4)
            self.CASF = CosineAnnealingScaleFactorScheduler(initial_sf=0.01, random_jump=False)
            # self.CASF = CosineAnnealingScaleFactorScheduler()
            # for i in range(1000):
            #     self.CASF.get_sf()
            self.min_delta = 1e-3
        else:
            self.initialized_planes = initialize_patches(self.rays_c,
                                                         self.ray_distances_c,
                                                         self.patch_vertexes_id)  # (num_patch, 4)
            self.CASF = CosineAnnealingScaleFactorScheduler()
            self.min_delta = 0.003

        # optimization data
        self.optimized_abcd_list = self.initialized_planes.clone()
        if "plane_vertex_pos_gt" in self.graph.graph:
            # plane_vertex_pos_gt: n*3*3 in world coordinate
            self.gt_abcd_list = self.compute_plane_pam_gt_in_cam(self.graph.graph["plane_vertex_pos_gt"])
            # self.optimized_abcd_list[0, :] = self.gt_abcd_list[9, :].clone()
            # self.optimized_abcd_list[1, :] = self.gt_abcd_list[3, :].clone()
            # self.optimized_abcd_list[2, :] = self.gt_abcd_list[9, :].clone()
            # self.gt_abcd_list = self.optimized_abcd_list.clone()
        else:
            self.gt_abcd_list = None

        self.patch_num = len(self.patches_list)
        self.cur_iter = [0] * self.patch_num
        self.end_flag = [0] * self.patch_num
        self.best_loss = [None] * self.patch_num
        self.delta = [None] * self.patch_num
        self.MAX_ITER = 1001
        self.MAX_TOLERENCE = 50
        self.num_tolerence = [self.MAX_TOLERENCE] * self.patch_num
        self.num_plane_sample = 100

        self.ncc_loss_weight = 10
        self.edge_loss_weight = 10
        self.reg_loss_weight = 0
        self.glue_loss_weight = 10
        self.regularization_loss_weight = 0
        self.mutex_loss_weight = 0
        self.edge_loss_computer = Edge_loss_computer(v_sample_density=0.001)
        self.glue_loss_computer = Glue_loss_computer(self.dual_graph, self.rays_c)
        self.regularization_loss_computer = Regularization_loss_computer(self.dual_graph)
        self.mutex_loss_computer = Mutex_loss_computer()
        self.ncc_loss_computer = Bilateral_ncc_computer()
        self.collision_checker = Collision_checker()

        self.id_neighbour_patches = [list(self.dual_graph[patch_id].keys()) for patch_id in self.dual_graph.nodes]

        # divide the patches into several groups
        self.use_group = use_group
        if use_group:
            # 1. partition the patches into several groups using the nx.greedy_color(dual_graph)
            patch_color = nx.greedy_color(self.dual_graph)
            groups = {}
            for key, value in patch_color.items():
                if value not in groups:
                    groups[value] = [key]
                else:
                    groups[value].append(key)
            self.groups = list(groups.values())
        else:
            self.groups = [list(range(self.patch_num))]
        self.group_idx = 0

        # vis
        self.visualizer = Visualizer(
                self.patch_num, self.v_log_root,
                self.imgs,
                self.intrinsic,
                self.extrinsic_ref_cam,
                self.transformation,
                debug_mode=False
                )
        self.visualizer.save_planes("init_plane.ply", self.initialized_planes, self.rays_c, self.patch_vertexes_id)

        self.visualizer.viz_results(-1, self.initialized_planes, self.rays_c, self.patch_vertexes_id)

        self.sample_g = torch.Generator(self.device)
        self.sample_g.manual_seed(0)

    def prepare_data(self, v_src_img_database: list[Image]):
        self.src_imgs = [cv2.imread(src_img.img_path, cv2.IMREAD_GRAYSCALE) for src_img in v_src_img_database]

        self.imgs = torch.from_numpy(
                np.concatenate(([self.ref_img], self.src_imgs), axis=0)).to(self.device).to(torch.float32) / 255.

        self.projection2 = np.stack([src_img.projection for src_img in v_src_img_database])

        # transformation store the transformation matrix from ref_img to src_imgs
        transformation = self.projection2 @ np.linalg.inv(self.ref_Image.extrinsic)
        self.transformation = torch.from_numpy(transformation).to(self.device).to(torch.float32)

        self.c1_2_c2_list = []
        for src_img in v_src_img_database:
            self.c1_2_c2_list.append(torch.from_numpy(
                    src_img.extrinsic @ np.linalg.inv(self.ref_Image.extrinsic)
                    ).to(self.device).to(torch.float32))

        self.camera_coor_in_c1 = torch.linalg.inv(torch.stack(self.c1_2_c2_list))[:, :3, -1]

        self.gradients2 = []
        for src_img in self.imgs[1::]:
            dy, dx = torch.gradient(src_img)
            gradients = torch.stack((dx, dy), dim=-1)
            # dilated_gradients = torch.from_numpy(dilate_edge(gradients)).to(device)
            # dilated_gradients_list.append(dilated_gradients)
            self.gradients2.append(gradients)
        self.gradients2 = torch.stack(self.gradients2)
        return

    def get_optimization_order(self):
        # sort the patches by the area of neighbour patches
        area_list = []
        for patch_id in range(self.patch_num):
            local_vertex_pos = intersection_of_ray_and_all_plane(self.optimized_abcd_list[patch_id].unsqueeze(0),
                                                                 self.rays_c[
                                                                     self.patch_vertexes_id[patch_id]])  # 100 * n * 3
            # 2. Construct the triangles
            num_vertex = len(self.patch_vertexes_id[patch_id])
            edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
            local_edge_pos = local_vertex_pos[:, edges_idx]
            local_centroid = intersection_of_ray_and_all_plane(self.optimized_abcd_list[patch_id].unsqueeze(0),
                                                               self.centroid_rays_c[patch_id].unsqueeze(0))[:, 0]
            triangles_pos = torch.cat((local_edge_pos, local_centroid[:, None, None, :].tile(1, num_vertex, 1, 1)),
                                      dim=2)
            triangles_pos = triangles_pos.view(-1, 3, 3)  # (1*num_tri,3,3)

            d1 = triangles_pos[:, 1, :] - triangles_pos[:, 0, :]
            d2 = triangles_pos[:, 2, :] - triangles_pos[:, 1, :]
            area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2
            area_list.append(area.sum().item())

        order_idx = sorted(torch.arange(self.patch_num).to(self.device), key=lambda x: area_list[x], reverse=True)
        order_idx = [idx.cpu().item() for idx in order_idx]
        return order_idx

    def reset(self):
        self.patch_num = len(self.patches_list)
        self.cur_iter = [0] * self.patch_num
        self.end_flag = [0] * self.patch_num
        self.best_loss = [None] * self.patch_num
        self.delta = [None] * self.patch_num
        self.MAX_ITER = 1001
        self.MAX_TOLERENCE = 50
        self.num_tolerence = [self.MAX_TOLERENCE] * self.patch_num
        self.num_plane_sample = 100

        self.sample_g = torch.Generator(self.device)
        self.sample_g.manual_seed(0)

    def run(self, only_optimize_patch_id=[]):
        self.reset()

        if only_optimize_patch_id:
            patch_id_list = only_optimize_patch_id
            for patch_id in range(self.patch_num):
                if patch_id not in only_optimize_patch_id:
                    self.end_flag[patch_id] = 1
        else:
            patch_id_list = self.get_optimization_order()

        sampler = Sample_new_planes(self.id_neighbour_patches, patch_id_list)

        while True:
            if all(self.end_flag):
                break

            if self.use_group and not only_optimize_patch_id:
                patch_id_list = [patch_id for patch_id in self.groups[self.group_idx] if not self.end_flag[patch_id]]
                self.group_idx = (self.group_idx + 1) % len(self.groups)
            else:
                patch_id_list = [patch_id for patch_id in patch_id_list]

            self.visualizer.start_iter()
            # sample for all patches
            samples_depth, samples_angle = sampler.sample(
                self.optimized_abcd_list, self.centroid_rays_c, self.CASF.get_sf(), self.sample_g, debug_gt=None, )
            # samples_depth, samples_angle = sample_new_planes(self.optimized_abcd_list,
            #                                                  self.centroid_rays_c,
            #                                                  self.id_neighbour_patches,
            #                                                  patch_id_list,
            #                                                  self.CASF.get_sf(),
            #                                                  v_random_g=self.sample_g,
            #                                                  debug_gt=None, )
            # self.gt_abcd_list[torch.tensor(patch_id_list)]

            samples_abc = angles_to_vectors(samples_angle)  # n * 100 * 3
            samples_intersection, samples_abcd = compute_plane_abcd(self.centroid_rays_c[patch_id_list], samples_depth,
                                                                    samples_abc)

            self.visualizer.update_timer("Sample")
            self.visualizer.update_sample_plane(patch_id_list, samples_abcd, self.rays_c, self.patch_vertexes_id,
                                                max(self.cur_iter))
            self.visualizer.update_timer("SampleVis")

            # have been checked, all end_flag is 0
            for idx, patch_id in enumerate(patch_id_list):
                self.optimize_patch(patch_id, samples_abcd[idx])

            self.update_triangle()

        self.save()
        return self.graph

    def optimize_patch(self, v_patch_id, local_plane_samples):
        # 1. Compute some share variables(vertex, edge, centroid) of this patch
        num_vertex = len(self.patch_vertexes_id[v_patch_id])
        # local_plane_samples: 100 * 4
        # local_vertex_pos： 100 * n * 3
        local_vertex_pos = intersection_of_ray_and_all_plane(local_plane_samples,
                                                             self.rays_c[self.patch_vertexes_id[v_patch_id]])
        self.visualizer.update_timer("Construct_p1")
        edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
        local_edge_pos = local_vertex_pos[:, edges_idx]
        local_centroid = intersection_of_ray_and_all_plane(local_plane_samples,
                                                           self.centroid_rays_c[v_patch_id].unsqueeze(0))[:, 0]

        self.visualizer.update_timer("Construct_p2")

        # 2. Construct the triangles
        sub_faces_id_c = self.dual_graph.nodes[int(v_patch_id)]['sub_faces_id']
        sub_faces_c = self.dual_graph.nodes[int(v_patch_id)]['sub_faces']
        triangles_pos_list = []
        triangles_num_per_face = []
        for sub_face_id, sub_face_vertices_id in zip(sub_faces_id_c, sub_faces_c):
            vertex_pos_sub_face = intersection_of_ray_and_all_plane(local_plane_samples,
                                                                    self.rays_c[sub_face_vertices_id])  # 100 * n * 3
            num_vertex_sf = len(sub_face_vertices_id)
            triangles_num_per_face.append(num_vertex_sf)
            indices = torch.arange(num_vertex_sf)
            edges_idx = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1).tolist()
            self.visualizer.update_timer("Construct_p3")
            edge_pos_sf = vertex_pos_sub_face[:, edges_idx]
            centroid_ray_sf = self.src_faces_centroid_rays_c[sub_face_id]
            centroid_sf = intersection_of_ray_and_all_plane(local_plane_samples,
                                                            centroid_ray_sf.unsqueeze(0))[:, 0]
            triangles_pos = torch.cat((edge_pos_sf, centroid_sf[:, None, None, :].tile(1, num_vertex_sf, 1, 1)),
                                      dim=2)
            triangles_pos_list.append(triangles_pos)  # (100,num_tri,3,3)
            self.visualizer.update_timer("Construct_p4")

        triangles_pos_per_sample = torch.cat(triangles_pos_list, dim=1)  # (100,num_tri,3,3)
        triangles_pos = triangles_pos_per_sample.view(-1, 3, 3)  # (100*num_tri,3,3)
        self.visualizer.update_timer("Construct")

        # 3. Sample points in this plane
        num_sample_points_per_tri, sample_points_on_face_src = sample_triangles(100,
                                                                                triangles_pos[:, 0, :],
                                                                                triangles_pos[:, 1, :],
                                                                                triangles_pos[:, 2, :],
                                                                                v_sample_edge=False,
                                                                                num_max_sample=100)

        triangle_normal = normalize_tensor(torch.cross(triangles_pos[:, 0, :] - triangles_pos[:, 1, :],
                                                       triangles_pos[:, 1, :] - triangles_pos[:, 2, :]))

        triangle_normal_src = triangle_normal.repeat_interleave(num_sample_points_per_tri, dim=0)
        points_to_tri_src = torch.arange(num_sample_points_per_tri.shape[0]).to(self.device).repeat_interleave(
                num_sample_points_per_tri)

        self.visualizer.update_timer("Sample points")

        # 1. Check collision
        # we need to check collision for each source image in the reference image coordinate
        # 1. get camera coordinate of the source image in the coordinate of the reference image
        # 2. trans the ray's origin to the source image camera pos in the reference image coordinate
        num_sp = sample_points_on_face_src.shape[0]
        ray_dir = sample_points_on_face_src[None, :] - self.camera_coor_in_c1[:, None]
        collision_flag = self.collision_checker.check_ray(
                v_patch_id,
                self.camera_coor_in_c1[:, None].tile(1, num_sp, 1).reshape(-1, 3),
                ray_dir.reshape(-1, 3),
                batch_size=1000000)
        collision_flag = collision_flag.reshape(-1, num_sp)
        remain_flag = torch.logical_not(collision_flag)
        # Skip this src_img if all points are in collision
        valid_img_flag = torch.any(remain_flag, dim=1)
        self.visualizer.update_timer("Collision")

        # 2. Compute loss
        # 1) NCC loss
        points_ncc = self.ncc_loss_computer.compute_batch(
                sample_points_on_face_src, triangle_normal_src,
                self.intrinsic, self.transformation, self.imgs[0], self.imgs[1:])
        # we do not consider loss of invisible points
        points_ncc[~remain_flag] = 0
        self.visualizer.update_timer("NCC1")

        num_source = len(self.src_imgs)
        points_to_tri = points_to_tri_src[None, :].tile(num_source, 1)
        triangles_ncc = scatter_sum(points_ncc, points_to_tri, dim=1)
        triangles_ncc /= scatter_sum(remain_flag.to(torch.long), points_to_tri, dim=1)
        triangles_ncc = triangles_ncc.view(num_source, self.num_plane_sample, -1)
        triangle_weights = num_sample_points_per_tri.view(self.num_plane_sample, -1)

        triangle_weights = triangle_weights / (triangle_weights.sum(dim=-1, keepdim=True) + 1e-6)
        ncc_loss = (triangles_ncc * triangle_weights).nanmean(dim=-1)
        self.visualizer.update_timer("NCC2")

        # 2) Edge loss
        edge_loss, edge_outer_mask, num_samples_per_edge = self.edge_loss_computer.compute_batch(
                local_edge_pos.reshape(-1, 2, 3),
                self.intrinsic,
                self.transformation,
                self.gradients1,
                self.gradients2,
                self.imgs[1:],
                v_num_hypothesis=local_edge_pos.shape[0],
                v_num_max_samples=100)
        edge_loss[~edge_outer_mask] = torch.inf

        edge_loss = edge_loss.view(num_source, self.num_plane_sample, -1)
        edge_loss = torch.nanmean(edge_loss, dim=2)

        self.visualizer.update_timer("Edge")

        # 3) Glue loss
        glue_loss = self.glue_loss_computer.compute(v_patch_id, self.optimized_abcd_list, local_vertex_pos)
        self.visualizer.update_timer("Glue")

        # weight
        ncc_loss *= torch.exp(-torch.pow(ncc_loss, 2) / 0.18)
        edge_loss *= torch.exp(-torch.pow(edge_loss, 2) / 0.18)

        final_loss = ncc_loss * self.ncc_loss_weight \
                     + edge_loss * self.edge_loss_weight \
                     + glue_loss * self.glue_loss_weight

        # 4. filter some loss
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
        assert outlier_mask.sum().item() != outlier_mask.shape[0]  # all the sample are outliers
        final_loss[outlier_mask] = torch.nan

        final_loss_sum = final_loss.nanmean(dim=0)
        # ncc_loss_sum = ncc_loss_sum.nanmean(dim=0)
        # edge_loss_sum = edge_loss_sum.nanmean(dim=0)
        self.visualizer.update_timer("Loss")

        # 5. Select the best and update `optimized_abcd_list`
        # all the sample of all the patch are collided
        # each sample should have at least 2 valid loss
        valid_loss_num = (~torch.isnan(final_loss)).sum(dim=0)
        final_loss_sum[torch.isnan(final_loss_sum)] = torch.inf

        id_best = torch.argmin(final_loss_sum, dim=-1)

        if self.best_loss[v_patch_id] is not None and valid_loss_num[id_best] < 2:
            id_best *= 0

        # if best_loss[v_patch_id] is not None and final_loss[id_best] < best_loss[v_patch_id]:
        self.optimized_abcd_list[v_patch_id] = local_plane_samples[id_best]

        # 6. Visualize
        self.visualizer.viz_patch_2d(
                v_patch_id,
                self.cur_iter[v_patch_id],

                local_plane_samples,
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
                self.end_flag[v_patch_id],
                )

        # 7. Control the end of the optimization
        cur_loss = final_loss_sum[id_best].cpu().item()
        if self.best_loss[v_patch_id] is not None:
            self.delta[v_patch_id] = cur_loss - self.best_loss[v_patch_id]
            print("Graph {} Patch {:3d} Iter {:4d} -> Loss: {:.4f} Best: {:.4f} Delta: {:+.4f} Id_Best: {:3d}".format(
                    self.ref_img_id,
                    v_patch_id,
                    self.cur_iter[v_patch_id] + 1,
                    cur_loss,
                    self.best_loss[v_patch_id],
                    self.delta[v_patch_id],
                    id_best.cpu().item())
                    )

            if abs(self.delta[v_patch_id]) * valid_loss_num[id_best] < self.min_delta:
                self.num_tolerence[v_patch_id] -= 1
            else:
                self.num_tolerence[v_patch_id] = self.MAX_TOLERENCE
                self.best_loss[v_patch_id] = cur_loss

            if self.num_tolerence[v_patch_id] <= 0:  # or best_loss[v_patch_id] < 0.000001
                self.end_flag[v_patch_id] = 1
        else:
            self.best_loss[v_patch_id] = cur_loss

        self.cur_iter[v_patch_id] += 1

        if self.cur_iter[v_patch_id] >= self.MAX_ITER:
            self.end_flag[v_patch_id] = 1

        self.visualizer.update_timer("Update1")
        return

    def update_triangle(self):
        # Update triangles
        final_triangles_list = []
        tri_num_each_patch = []
        for v_patch_id in range(self.patch_num):
            sub_faces_id_c = self.dual_graph.nodes[int(v_patch_id)]['sub_faces_id']
            sub_faces_c = self.dual_graph.nodes[int(v_patch_id)]['sub_faces']
            triangles_pos_list = []
            triangles_num_per_face = []
            for sub_face_id, sub_face_vertices_id in zip(sub_faces_id_c, sub_faces_c):
                vertex_pos_sub_face = intersection_of_ray_and_all_plane(
                        self.optimized_abcd_list[v_patch_id:v_patch_id + 1],
                        self.rays_c[sub_face_vertices_id])  # 100 * n * 3
                num_vertex_sf = len(sub_face_vertices_id)
                triangles_num_per_face.append(num_vertex_sf)
                indices = torch.arange(num_vertex_sf)
                edges_idx = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1).tolist()
                edge_pos_sf = vertex_pos_sub_face[:, edges_idx]
                centroid_ray_sf = self.src_faces_centroid_rays_c[sub_face_id]
                centroid_sf = intersection_of_ray_and_all_plane(self.optimized_abcd_list[v_patch_id:v_patch_id + 1],
                                                                centroid_ray_sf.unsqueeze(0))[:, 0]
                triangles_pos = torch.cat(
                        (edge_pos_sf, centroid_sf[:, None, None, :].tile(1, num_vertex_sf, 1, 1)),
                        dim=2)
                triangles_pos_list.append(triangles_pos)  # (1,num_tri,3,3)

            triangles_pos_per_sample = torch.cat(triangles_pos_list, dim=1)  # (1,num_tri,3,3)
            tri_num_each_patch.append(triangles_pos_per_sample.shape[1])
            final_triangles_list.append(triangles_pos_per_sample[0])

        final_triangles = torch.cat(final_triangles_list)
        tri_to_patch = torch.arange(self.patch_num).repeat_interleave(
                torch.from_numpy(np.array(tri_num_each_patch))).to(
                self.device)

        self.collision_checker.clear()
        self.collision_checker.add_triangles(final_triangles, tri_to_patch)

        self.visualizer.update_timer("Update2")
        self.visualizer.viz_results(
                max(self.cur_iter),
                self.optimized_abcd_list, self.rays_c, self.patch_vertexes_id
                )
        return

    def eval(self, optimized_abcd_tensor, gt_abcd_tensor):
        optimized_abcd_tensor_norm = optimized_abcd_tensor / torch.norm(optimized_abcd_tensor, dim=1, keepdim=True)
        gt_abcd_tensor_norm = gt_abcd_tensor / torch.norm(gt_abcd_tensor, dim=1, keepdim=True)

        gt_abcd_tensor_norm_combined = torch.cat((gt_abcd_tensor_norm, -gt_abcd_tensor_norm), dim=0)

        distances = torch.cdist(optimized_abcd_tensor_norm, gt_abcd_tensor_norm_combined)

        closest_gt_indices = torch.argmin(distances, dim=1)
        closest_gt_indices = closest_gt_indices % len(gt_abcd_tensor_norm)

        differences = torch.min(distances, dim=1).values

        return closest_gt_indices, differences

    def save(self):
        self.visualizer.save_planes("optimized.ply", self.optimized_abcd_list, self.rays_c, self.patch_vertexes_id)
        self.visualizer.save_planes(str(self.ref_img_id) + ".ply", self.optimized_abcd_list, self.rays_c,
                                    self.patch_vertexes_id, transform_to_world=True,
                                    file_dir=os.path.join(os.path.dirname(self.v_log_root), "optimized_world"))

        for idx, loss in enumerate(self.best_loss):
            self.dual_graph.nodes[idx]['loss'] = loss

        np.save(os.path.join(self.v_log_root, "optimized_graph_c.npy"), np.asarray([self.graph], dtype=object),
                allow_pickle=True)

        self.dual_graph.graph['optimized_abcd_list'] = self.optimized_abcd_list

        print("\nGraph {} optimization End. Cost time: {}s\n".format(self.ref_img_id,
                                                                     self.visualizer.time_consume()))

        if self.gt_abcd_list is not None:
            closest_gt_indices, differences = self.eval(self.optimized_abcd_list, self.gt_abcd_list)

            # 打开一个文件用于写入
            with open(os.path.join(self.v_log_root, 'evaluation_results.txt'), 'w') as file:
                file.write(f"Graph {self.ref_img_id} optimization, Cost time {self.visualizer.time_consume()}s\n")
                for i, (index, diff) in enumerate(zip(closest_gt_indices, differences)):
                    result_str = f"Optimized plane {i} is closest to ground truth plane {index} with a difference of {diff:.4f}\n"
                    print(result_str, end='')
                    file.write(result_str)

        gc.collect()
        torch.cuda.empty_cache()
        return

    def compute_plane_pam_gt_in_cam(self, plane_vertex_pos_gt):
        # plane_vertex_pos_gt: n * 3 * 3 in world coordinate
        plane_pam_gt = []
        for plane_vertex_pos_c in plane_vertex_pos_gt:
            # trans to cam coordinate
            plane_vertex_pos_c = (self.extrinsic_ref_cam @ to_homogeneous_tensor(plane_vertex_pos_c).T).T
            plane_vertex_pos_c = plane_vertex_pos_c[:, :3] / (plane_vertex_pos_c[:, 3:4] + 1e-8)

            edge1 = plane_vertex_pos_c[1] - plane_vertex_pos_c[0]
            edge2 = plane_vertex_pos_c[2] - plane_vertex_pos_c[0]
            normal = torch.cross(edge1, edge2)
            d = -torch.dot(normal, plane_vertex_pos_c[0])
            plane_parameters = torch.cat((normal, d.unsqueeze(0)))

            plane_pam_gt.append(plane_parameters)
        return torch.stack(plane_pam_gt)
