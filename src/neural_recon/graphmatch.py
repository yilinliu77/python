import networkx as nx
import torch
import numpy as np
from src.neural_recon.geometric_util import angles_to_vectors, compute_plane_abcd, intersection_of_ray_and_all_plane, \
    intersection_of_ray_and_plane
import cv2
from shared.common_utils import normalize_tensor, normalize_vector, to_homogeneous_tensor
from src.neural_recon.geometric_util import fit_plane_svd, point_to_plane_distance
from src.neural_recon.sample_utils import sample_triangles
import random
import os
import open3d as o3d
import warnings
from src.neural_recon.GraphPlaneOptimization import GraphPlaneOptimiser


def merge_matches(matches):
    def find(_parent, i):
        if _parent[i] != i:
            _parent[i] = find(_parent, _parent[i])
        return _parent[i]

    def union(_parent, _rank, _x, _y):
        xroot = find(_parent, _x)
        yroot = find(_parent, _y)
        if _rank[xroot] < _rank[yroot]:
            _parent[xroot] = yroot
        elif _rank[xroot] > _rank[yroot]:
            _parent[yroot] = xroot
        else:
            _parent[yroot] = xroot
            _rank[xroot] += 1

    # Initialize parent and rank dictionaries
    parent = {item: item for match in matches for item in match}
    rank = {item: 0 for match in matches for item in match}

    # Perform union operations for each match
    for x, y in matches:
        union(parent, rank, x, y)

    # Find the root for each item and group them
    groups = {}
    for item in parent:
        root = find(parent, item)
        if root not in groups:
            groups[root] = []
        groups[root].append(item)

    # Convert the groups dictionary to a list of groups
    return list(groups.values())


# 1.完成local rencon的匹配与融合
# 2.完成基于全局一致性图的点优化和线优化
class PairGraphMatcher:
    def __init__(self, graph_ref, graph_src, v_log_root, device, tolerance_3d=0.1, tolerance_2d=0.01,
                 distance_threshold=0.1, inlier_rate_threshold=0.8, normal_angle_threshold=10, is_exact_match=False):
        self.graph_ref = graph_ref
        self.graph_src = graph_src
        self.v_img_ref = graph_ref.graph["v_img"]
        self.v_img_src = graph_src.graph["v_img"]

        self.ref_img = cv2.imread(self.v_img_ref.img_path)
        self.src_img = cv2.imread(self.v_img_src.img_path)
        self.ref_img_grey = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.src_img_grey = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        self.ref_img_tensor = torch.from_numpy(self.ref_img_grey).to(device).to(torch.float32) / 255.
        self.src_img_tensor = torch.from_numpy(self.src_img_grey).to(device).to(torch.float32) / 255.

        self.tf_ref2src = torch.from_numpy(self.v_img_src.projection @ np.linalg.inv(self.v_img_ref.extrinsic)
                                           ).to(device).to(torch.float32)
        self.tf_src2ref = torch.from_numpy(self.v_img_ref.projection @ np.linalg.inv(self.v_img_src.extrinsic)
                                           ).to(device).to(torch.float32)
        self.c_ref2src = torch.from_numpy(self.v_img_src.extrinsic @ np.linalg.inv(self.v_img_ref.extrinsic)
                                          ).to(device).to(torch.float32)
        self.c_src2ref = torch.from_numpy(self.v_img_ref.extrinsic @ np.linalg.inv(self.v_img_src.extrinsic)
                                          ).to(device).to(torch.float32)

        self.rays_ref = torch.from_numpy(np.stack([graph_ref.nodes[i]["ray_c"] for i in graph_ref.nodes])).to(
                device).to(torch.float32)
        self.rays_src = torch.from_numpy(np.stack([graph_src.nodes[i]["ray_c"] for i in graph_src.nodes])).to(
                device).to(torch.float32)

        self.sub_faces_centroid_rays_ref = torch.from_numpy(graph_ref.graph["src_face_ray_c"]).to(device)
        self.sub_faces_centroid_rays_src = torch.from_numpy(graph_src.graph["src_face_ray_c"]).to(device)

        self.dual_graph_ref = graph_ref.graph["dual_graph"]
        self.dual_graph_src = graph_src.graph["dual_graph"]

        self.patches_ref = [graph_ref.graph["dual_graph"].nodes[i]["id_vertex"] for i in
                            graph_ref.graph["dual_graph"].nodes]
        self.patches_src = [graph_src.graph["dual_graph"].nodes[i]["id_vertex"] for i in
                            graph_src.graph["dual_graph"].nodes]

        self.patches_center_ref = [graph_ref.graph["dual_graph"].nodes[i]["face_center"] for i in
                                   graph_ref.graph["dual_graph"].nodes]
        self.patches_center_src = [graph_src.graph["dual_graph"].nodes[i]["face_center"] for i in
                                   graph_src.graph["dual_graph"].nodes]

        self.optimized_abcd_list_ref = graph_ref.graph["dual_graph"].graph['optimized_abcd_list']
        self.optimized_abcd_list_src = graph_src.graph["dual_graph"].graph['optimized_abcd_list']

        self.intrinsic = torch.from_numpy(self.v_img_ref.intrinsic).to(device).to(torch.float32)
        self.transformation_w2c_ref = torch.from_numpy(self.v_img_ref.extrinsic).to(device).to(torch.float32)
        self.transformation_c2w_ref = torch.linalg.inv(self.transformation_w2c_ref)
        self.transformation_w2c_src = torch.from_numpy(self.v_img_src.extrinsic).to(device).to(torch.float32)
        self.transformation_c2w_src = torch.linalg.inv(self.transformation_w2c_src)
        self.device = device

        # prepare the data for matching
        self.tolerance_3d = tolerance_3d
        self.tolerance_2d = tolerance_2d
        self.distance_threshold = distance_threshold
        self.inlier_rate_threshold = inlier_rate_threshold
        self.normal_angle_threshold = normal_angle_threshold
        self.is_exact_match = is_exact_match

        if self.is_exact_match:
            self.distance_threshold = 0.05
            self.inlier_rate_threshold = 0.99
            self.normal_angle_threshold = 1
        else:
            self.distance_threshold = 0.1
            self.inlier_rate_threshold = 0.8
            self.normal_angle_threshold = 10

        self.v_log_root = v_log_root
        os.makedirs(self.v_log_root, exist_ok=True)
        self.recon_graph_ref = None
        self.recon_graph_src = None

        self.recon_graph_matches_node = []
        self.recon_graph_matches_face = []
        self.match_points = {}
        self.merged_graph = None
        self.merged_graph_pointset = None
        self.invalid_graph = []

    def run(self):
        print("PairGraphMatcher Beginning: Graph {} and Graph {}".format(self.graph_ref.name, self.graph_src.name))
        # 1. create the recon_graph_ref and recon_graph_src
        if self.recon_graph_ref is None:
            self.recon_graph_ref = self.create_graph(self.graph_ref, self.patches_ref, self.rays_ref,
                                                     self.optimized_abcd_list_ref, self.transformation_c2w_ref,
                                                     self.c_ref2src, self.tf_ref2src)
            self.recon_graph_src = self.create_graph(self.graph_src, self.patches_src, self.rays_src,
                                                     self.optimized_abcd_list_src, self.transformation_c2w_src,
                                                     self.c_src2ref, self.tf_src2ref)
        if self.invalid_graph:
            print("Failed to create recon_graph_ref or recon_graph_src, invalid graph: {}".format(self.invalid_graph))
            return None
        else:
            self.save_wireframe(self.recon_graph_ref, self.graph_ref.name, "recon_graph_ref.obj")
            self.save_wireframe(self.recon_graph_src, self.graph_src.name, "recon_graph_src.obj")

        # 2. find match nodes of recon_graph_ref and recon_graph_src
        self.recon_graph_matches_node = self.match_node()
        self.recon_graph_matches_face = self.match_face()

        # 3. merge the recon_graph_ref and recon_graph_src (in recon_graph_ref's camera coordinate)
        if self.recon_graph_matches_node:
            self.merge()

        print("PairGraphMatcher: Graph {} and Graph {} have {} matches nodes, {} matches faces\n".format(
                self.graph_ref.name,
                self.graph_src.name,
                len(self.recon_graph_matches_node),
                len(self.recon_graph_matches_face)))

        print("PairGraphMatcher End: Graph {} and Graph {}\n".format(self.graph_ref.name, self.graph_src.name))
        return

    def create_graph(self, graph, patches, rays, optimized_abcd_list, tf_c2w, tf_camera=None, tf_img=None):
        recon_graph = nx.Graph()
        recon_graph.name = graph.name

        # checking the vaildity of the patches
        patch_loss = [graph.graph['dual_graph'].nodes[i]['loss'] for i in graph.graph['dual_graph'].nodes]
        mean_patch_loss = sum(patch_loss) / len(patch_loss)
        isvalid_patch = [loss < 0.5 for loss in patch_loss]
        # isvalid_patch = [True for loss in patch_loss]
        if not sum(isvalid_patch):
            self.invalid_graph.append(graph.name)
            print("\033[91mError: Checking failed! No valid patches in graph {}\033[0m".format(graph.name))
            return None
        elif sum(isvalid_patch) < len(isvalid_patch):
            print("\033[93mWarning: Filtering {} invalid patches in graph {}\033[0m".format(
                    len(isvalid_patch) - sum(isvalid_patch),
                    graph.name))

        patches_new_name = []
        patches_pos_world = []
        patches_pos_2d = []
        for patch_idx, patch_c in enumerate(patches):
            if not isvalid_patch[patch_idx]:
                patches_new_name.append(None)
                patches_pos_world.append(None)
                patches_pos_2d.append(None)
                continue
            # add nodes, the names of nodes are vertex_id of graph
            pos_3d = intersection_of_ray_and_all_plane(optimized_abcd_list[patch_idx].unsqueeze(0), rays[patch_c])[0]
            pos_3d_tf = (tf_camera @ to_homogeneous_tensor(pos_3d).T).T
            pos_3d_tf = pos_3d_tf[:, :3] / (pos_3d_tf[:, 3:4] + 1e-8)

            pos_2d = (self.intrinsic @ pos_3d.T).T
            pos_2d = pos_2d[:, :2] / (pos_2d[:, 2:3] + 1e-8)
            pos_2d_ = np.array([graph.nodes[f'G{graph.name}_{vertex_id}']["pos_2d"] for vertex_id in patch_c],
                               dtype=np.float32)
            pos_2d_ = torch.from_numpy(pos_2d_).to(self.device).to(torch.float32)

            pos_2d_tf = (tf_img @ to_homogeneous_tensor(pos_3d).T).T
            pos_2d_tf = pos_2d_tf[:, :2] / (pos_2d_tf[:, 2:3] + 1e-8)
            valid_mask_tf = torch.logical_and(pos_2d_tf < 1, pos_2d_tf > 0).all(dim=-1)

            # pos_3d in world
            pos_3d_world = (tf_c2w @ to_homogeneous_tensor(pos_3d).T).T
            pos_3d_world = pos_3d_world[:, :3] / (pos_3d_world[:, 3:4] + 1e-8)

            patch_c_new_name = []
            for idx, (pos_3d_, pos_2d_, pos_3d_tf_, pos_2d_tf_, pos_3d_world_) in enumerate(
                    zip(pos_3d, pos_2d, pos_3d_tf, pos_2d_tf, pos_3d_world)):
                node_name = f'G{graph.name}_{patch_c[idx]}'
                patch_c_new_name.append(node_name)
                recon_graph.add_node(node_name, graph_name=graph.name,
                                     patch_id=patch_idx, vertex_id=patch_c[idx],
                                     pos_3d_world=pos_3d_world_, pos_3d=pos_3d_, pos_2d=pos_2d_,
                                     pos_3d_tf=pos_3d_tf_, pos_2d_tf=pos_2d_tf_, is_valid_tf=valid_mask_tf[idx])

            id_edge_per_patch = [(f'G{graph.name}_{patch_c[idx - 1]}', f'G{graph.name}_{patch_c[idx]}')
                                 for idx in range(len(patch_c))]
            recon_graph.add_edges_from(id_edge_per_patch)

            patches_new_name.append(patch_c_new_name)
            patches_pos_world.append(pos_3d_world)
            patches_pos_2d.append(pos_2d)

        recon_graph.graph['patches'] = patches_new_name
        recon_graph.graph['patches_pos_world'] = patches_pos_world
        recon_graph.graph['patches_pos_2d'] = patches_pos_2d

        return recon_graph

    # need to fix
    def create_pointset_graph(self, graph):
        # resave the merged_graph's nodes attributes using PointSet
        merged_graph_pointset = merged_graph.copy()
        for node, data in merged_graph_pointset.nodes(data=True):
            if data['graph_name'] == self.graph_ref.name:
                pos = data['pos_3d']
            else:
                pos = data['pos_3d_tf']
            new_data = PointSet(pos, data['graph_name'], data['patch_id'], data['vertex_id'])
            merged_graph_pointset.nodes[node].clear()
            merged_graph_pointset.nodes[node]['data'] = new_data
        self.merged_graph_pointset = merged_graph_pointset

    def is_node_match(self, n1_attrs, n2_attrs):
        pos_2d_1 = n1_attrs['pos_2d']
        pos_2d_1_tf = n1_attrs['pos_2d_tf']

        pos_2d_2 = n2_attrs['pos_2d']
        pos_2d_2_tf = n2_attrs['pos_2d_tf']

        pos_3d_1 = n1_attrs['pos_3d_tf']
        pos_3d_2 = n2_attrs['pos_3d']

        return (torch.linalg.norm(pos_3d_1 - pos_3d_2) < self.tolerance_3d
                and torch.linalg.norm(pos_2d_1_tf - pos_2d_2) < self.tolerance_2d
                and torch.linalg.norm(pos_2d_1 - pos_2d_2_tf) < self.tolerance_2d)

    def is_face_match(self, patch_id_ref, patch_id_src, pos_world_ref, pos_world_src, is_vis=True):
        def sample_point_in_plane(dual_graph, sub_faces_centroid_rays_c, v_rays_c, v_patch_id,
                                  local_plane_parameter, transformation_c2w):
            # 1. Construct the triangles
            sub_faces_id_c = dual_graph.nodes[int(v_patch_id)]['sub_faces_id']
            sub_faces_c = dual_graph.nodes[int(v_patch_id)]['sub_faces']
            triangles_pos_list = []
            triangles_num_per_face = []
            for sub_face_id, sub_face_vertices_id in zip(sub_faces_id_c, sub_faces_c):
                vertex_pos_sub_face = intersection_of_ray_and_all_plane(local_plane_parameter.unsqueeze(0),
                                                                        v_rays_c[sub_face_vertices_id])  # 1 * n * 3
                num_vertex_sf = len(sub_face_vertices_id)
                triangles_num_per_face.append(num_vertex_sf)
                indices = torch.arange(num_vertex_sf)
                edges_idx = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1).tolist()

                edge_pos_sf = vertex_pos_sub_face[:, edges_idx]
                centroid_ray_sf = sub_faces_centroid_rays_c[sub_face_id]
                centroid_sf = intersection_of_ray_and_all_plane(local_plane_parameter.unsqueeze(0),
                                                                centroid_ray_sf.unsqueeze(0))[:, 0]
                triangles_pos = torch.cat((edge_pos_sf, centroid_sf[:, None, None, :].tile(1, num_vertex_sf, 1, 1)),
                                          dim=2)
                triangles_pos_list.append(triangles_pos)  # (1,num_tri,3,3)

            triangles_pos_per_sample = torch.cat(triangles_pos_list, dim=1)  # (1,num_tri,3,3)
            triangles_pos = triangles_pos_per_sample.view(-1, 3, 3)  # (num_tri,3,3)

            # 2. Sample points in those triangles
            _, sample_points_on_face = sample_triangles(100000,
                                                        triangles_pos[:, 0, :],
                                                        triangles_pos[:, 1, :],
                                                        triangles_pos[:, 2, :],
                                                        num_max_sample=10000,
                                                        v_sample_edge=True)
            # proj to world
            sample_points_on_face = (transformation_c2w @ to_homogeneous_tensor(sample_points_on_face).T).T
            sample_points_on_face = sample_points_on_face[:, :3] / (sample_points_on_face[:, 3:4] + 1e-8)
            return sample_points_on_face

        def bidirectional_distance1(sample_points1, sample_points2, dis_threshold):
            distances = torch.cdist(sample_points1, sample_points2, p=2)
            dis_1_to_2, _ = torch.min(distances, dim=0)
            dis_2_to_1, _ = torch.min(distances, dim=1)
            return min(dis_1_to_2.mean(), dis_2_to_1.mean())
            # return (dis_to_1.mean() + dis_to_2.mean()) / 2

        def bidirectional_distance2(sample_points1, sample_points2, dis_threshold):
            distances = torch.cdist(sample_points1, sample_points2, p=2)
            dis_to_1, _ = torch.min(distances, dim=0)
            dis_to_2, _ = torch.min(distances, dim=1)
            inlier_rate1 = (dis_to_1 < dis_threshold).sum() / dis_to_1.shape[0]
            inlier_rate2 = (dis_to_2 < dis_threshold).sum() / dis_to_2.shape[0]
            return (inlier_rate1 + inlier_rate2) / 2

        def hausdorff_distance(sample_points1, sample_points2):
            distances = torch.cdist(sample_points1, sample_points2, p=2)
            dis_1_to_2, _ = torch.min(distances, dim=1)
            dis_2_to_1, _ = torch.min(distances, dim=0)
            return max(dis_1_to_2.max(), dis_2_to_1.max())

        def compute_normal_angle(n1, n2):
            cos_theta = torch.clamp(torch.dot(n1, n2) / (torch.linalg.norm(n1) * torch.linalg.norm(n2)), -1, 1)
            theta = torch.acos(cos_theta)
            theta_degree = (theta * 180 / np.pi).item()
            return min(theta_degree, 180 - theta_degree)

        sample_point_in_plane_ref = sample_point_in_plane(self.dual_graph_ref, self.sub_faces_centroid_rays_ref,
                                                          self.rays_ref, patch_id_ref,
                                                          self.optimized_abcd_list_ref[patch_id_ref],
                                                          self.transformation_c2w_ref)
        sample_point_in_plane_src = sample_point_in_plane(self.dual_graph_src, self.sub_faces_centroid_rays_src,
                                                          self.rays_src, patch_id_src,
                                                          self.optimized_abcd_list_src[patch_id_src],
                                                          self.transformation_c2w_src)

        plane_abcd_ref = fit_plane_svd(pos_world_ref)
        plane_abcd_src = fit_plane_svd(pos_world_src)

        if self.is_exact_match:
            # inlier_rate = bidirectional_distance2(sample_point_in_plane_ref, sample_point_in_plane_src,
            #                                       self.distance_threshold)
            # normal_angle = compute_normal_angle(plane_abcd_ref[:3], plane_abcd_src[:3])
            # is_match = ((inlier_rate > self.inlier_rate_threshold) and (normal_angle < self.normal_angle_threshold))

            h_dis = hausdorff_distance(sample_point_in_plane_ref, sample_point_in_plane_src)
            normal_angle = compute_normal_angle(plane_abcd_ref[:3], plane_abcd_src[:3])
            is_match = ((h_dis < self.distance_threshold) and (normal_angle < self.normal_angle_threshold))

        else:
            b_dis = bidirectional_distance1(sample_point_in_plane_ref, sample_point_in_plane_src,
                                            self.distance_threshold)
            normal_angle = compute_normal_angle(plane_abcd_ref[:3], plane_abcd_src[:3])
            is_match = ((b_dis < self.distance_threshold) and (normal_angle < self.normal_angle_threshold))

        if is_vis:
            def torch_to_o3d(tensor):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(tensor.cpu().numpy())
                return pcd

            pcd1 = torch_to_o3d(sample_point_in_plane_ref)
            pcd2 = torch_to_o3d(sample_point_in_plane_src)
            pcd1.paint_uniform_color([1, 0, 0])
            pcd2.paint_uniform_color([0, 0, 1])
            print("Bidirectional distance:", b_dis, "Normal angle:", normal_angle)
            o3d.visualization.draw_geometries([pcd1, pcd2])

            if is_match:
                match = (f'G{self.recon_graph_ref.name}_P{patch_id_ref}',
                         f'G{self.recon_graph_src.name}_P{patch_id_src}')
            else:
                match = (f'NG{self.recon_graph_ref.name}_P{patch_id_ref}',
                         f'NG{self.recon_graph_src.name}_P{patch_id_src}')
            self.vis_match_face([match], [(sample_point_in_plane_ref, sample_point_in_plane_src)], is_show_window=True)

        return is_match, [sample_point_in_plane_ref, sample_point_in_plane_src]

    def match_node(self, is_vis_process=False, is_vis_result=True):
        matches = []
        recon_graph_ref_nodes = list(self.recon_graph_ref.nodes(data=True))
        recon_graph_src_nodes = list(self.recon_graph_src.nodes(data=True))
        matched_nodes_src = []

        checking_num = 0
        for n1, n1_attrs in recon_graph_ref_nodes:
            checking_num += 1
            if not n1_attrs['is_valid_tf']:
                continue
            for n2, n2_attrs in recon_graph_src_nodes:
                if n2 in matched_nodes_src:
                    continue
                if is_vis_process:
                    self.vis_match([(n1, n2)], is_show_window=True, is_save_file=False)
                if self.is_node_match(n1_attrs, n2_attrs):
                    matched_nodes_src.append(n2)
                    matches.append((n1, n2))
                    break

        if is_vis_result:
            self.vis_match(matches)

        return matches

    def match_face(self, is_vis_process=False, is_save_result=True):
        matches = []

        patches_ref = self.recon_graph_ref.graph['patches']
        patches_src = self.recon_graph_src.graph['patches']
        patches_pos_world_ref = self.recon_graph_ref.graph['patches_pos_world']
        patches_pos_world_src = self.recon_graph_src.graph['patches_pos_world']

        all_matches_result = []
        sample_points = []
        for idx_ref, patch_ref in enumerate(patches_ref):
            for idx_src, patch_src in enumerate(patches_src):
                # for those filtered patch, it will be set 'None' in patches_ref
                # so here the idx = real patch idx in dual graph
                if not patch_ref or not patch_src:
                    continue
                is_match, sample_points_c = self.is_face_match(idx_ref, idx_src, patches_pos_world_ref[idx_ref],
                                                               patches_pos_world_src[idx_src], is_vis_process)
                sample_points.append(sample_points_c)
                if is_match:
                    match = (f'G{self.recon_graph_ref.name}_P{idx_ref}', f'G{self.recon_graph_src.name}_P{idx_src}')
                    matches.append(match)
                    all_matches_result.append(match)
                    if sample_points_c[0].shape[0] < sample_points_c[1].shape[0]:
                        self.match_points[idx_ref] = sample_points_c[0]
                    else:
                        self.match_points[idx_src] = sample_points_c[1]
                    break
                else:
                    match = (f'NG{self.recon_graph_ref.name}_P{idx_ref}', f'NG{self.recon_graph_src.name}_P{idx_src}')
                    all_matches_result.append(match)

        if is_save_result:
            self.vis_match_face(all_matches_result, sample_points, is_save_file=True)
            if self.match_points:
                pc = torch.cat(list(self.match_points.values())).cpu().numpy()
                file_path = os.path.join(self.v_log_root, "matched_point_cloud.ply")
                self.save_point_cloud(pc, file_path)
                file_path = os.path.join(os.path.dirname(self.v_log_root),
                                         "matched_point_cloud",
                                         str(self.graph_ref.name) + "_" + str(self.graph_src.name) + ".ply")
                self.save_point_cloud(pc, file_path)

        return matches

    def merge(self, matches=None):
        if not matches:
            matches = self.recon_graph_matches_node

        recon_graph_ref = self.recon_graph_ref.copy()
        recon_graph_src = self.recon_graph_src.copy()

        matching_mapping = {ref: src for ref, src in matches}
        recon_graph_ref = nx.relabel_nodes(recon_graph_ref, matching_mapping, copy=False)

        # for the same label node, nx.compose will use the node attribute in the second graph
        merged_graph = nx.compose(recon_graph_ref, recon_graph_src)
        self.merged_graph = merged_graph
        self.save_wireframe(merged_graph, self.graph_ref.name, "merged.obj")
        out_path = os.path.join(os.path.dirname(self.v_log_root),
                                "merged_obj",
                                str(self.graph_ref.name) + "_" + str(self.graph_src.name) + ".obj")
        self.save_wireframe(merged_graph, self.graph_ref.name, out_path=out_path)
        return merged_graph

    def vis_match(self, matches, is_show_window=False, is_save_file=True):
        img_ref = self.ref_img.copy()
        img_src = self.src_img.copy()
        img_ref_grey = self.ref_img_grey.copy()
        img_src_grey = self.src_img_grey.copy()

        # draw all the checking nodes
        img_ref_copy = img_ref.copy()
        for n1, n1_attrs in self.recon_graph_ref.nodes(data=True):
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            pos_2d = (n1_attrs["pos_2d"].cpu().numpy() * img_ref_grey.shape[::1]).astype(np.int32)
            cv2.drawMarker(img_ref_copy, (pos_2d[0], pos_2d[1]), color=random_color, thickness=1,
                           markerType=cv2.MARKER_DIAMOND, markerSize=15)
        img_src_copy = img_src.copy()
        for n1, n1_attrs in self.recon_graph_src.nodes(data=True):
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            pos_2d = (n1_attrs["pos_2d"].cpu().numpy() * img_src_grey.shape[::1]).astype(np.int32)
            cv2.drawMarker(img_src_copy, (pos_2d[0], pos_2d[1]), color=random_color, thickness=1,
                           markerType=cv2.MARKER_DIAMOND, markerSize=15)
        checking_nodes_img = np.hstack((img_ref_copy, img_src_copy))

        # vis the matches
        outImg = np.hstack((img_ref.copy(), img_src.copy()))
        for n1, n2 in matches:
            pos_2d_n1_src = (self.graph_ref.nodes[n1]["pos_2d"] * img_ref_grey.shape[::1]).astype(np.int32)
            pos_2d_n2_src = (self.graph_src.nodes[n2]["pos_2d"] * img_src_grey.shape[::1]).astype(np.int32)

            pos_2d_n1 = (self.recon_graph_ref.nodes[n1]["pos_2d"].cpu().numpy() * img_ref_grey.shape[::1]).astype(
                    np.int32)
            pos_2d_n2 = (self.recon_graph_src.nodes[n2]["pos_2d"].cpu().numpy() * img_src_grey.shape[::1]).astype(
                    np.int32)

            pos_2d_tf_n1 = (self.recon_graph_ref.nodes[n1]["pos_2d_tf"].cpu().numpy() * img_ref_grey.shape[::1]).astype(
                    np.int32)
            pos_2d_tf_n2 = (self.recon_graph_src.nodes[n2]["pos_2d_tf"].cpu().numpy() * img_src_grey.shape[::1]).astype(
                    np.int32)

            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(outImg, (pos_2d_n1[0], pos_2d_n1[1]), (pos_2d_n2[0] + img_ref.shape[1], pos_2d_n2[1]),
                     random_color, thickness=1)
            # cv2.circle(outImg, (pos_2d_n1[0], pos_2d_n1[1]), radius=6, color=random_color, thickness=1)
            # cv2.circle(outImg, (pos_2d_n2[0] + img_ref.shape[1], pos_2d_n2[1]), radius=6, color=random_color,
            #            thickness=1)
            cv2.drawMarker(outImg, (pos_2d_n1[0], pos_2d_n1[1]), color=random_color, thickness=1,
                           markerType=cv2.MARKER_DIAMOND, markerSize=15)
            cv2.drawMarker(outImg, (pos_2d_n2[0] + img_ref.shape[1], pos_2d_n2[1]), color=random_color,
                           thickness=1, markerType=cv2.MARKER_DIAMOND, markerSize=15)
            cv2.drawMarker(outImg, (pos_2d_tf_n1[0] + img_ref.shape[1], pos_2d_tf_n1[1]), color=random_color,
                           thickness=1, markerType=cv2.MARKER_CROSS, markerSize=15)

        outImg = np.vstack((checking_nodes_img, outImg))
        if is_show_window:
            cv2.imshow("matches", outImg)
            cv2.waitKey(0)
        if is_save_file:
            cv2.imwrite(os.path.join(self.v_log_root, "matches.png"), outImg)
        return

    def vis_match_face(self, matches, sample_points, is_show_window=False, is_save_file=False):
        img_ref = self.ref_img.copy()
        img_src = self.src_img.copy()
        img_ref_grey = self.ref_img_grey.copy()
        img_src_grey = self.src_img_grey.copy()

        # vis the matches
        outImg = np.hstack((img_ref, img_src))
        for idx, (patch1, patch2) in enumerate(matches):
            # record patch1[0] == "N" is used to show online window for each checking pair (even though they are not matched) and debug
            if not is_show_window and patch1[0] == "N":
                continue
            patch_idx1 = int(patch1.split("_")[-1][1:])
            patch_idx2 = int(patch2.split("_")[-1][1:])
            patch_center1 = self.patches_center_ref[patch_idx1]
            patch_center2 = self.patches_center_src[patch_idx2]

            pos_3d_world1 = self.recon_graph_ref.graph['patches_pos_world'][patch_idx1]
            pos_3d_world2 = self.recon_graph_src.graph['patches_pos_world'][patch_idx2]

            pos_2d_1 = self.recon_graph_ref.graph['patches_pos_2d'][patch_idx1]
            pos_2d_2 = self.recon_graph_src.graph['patches_pos_2d'][patch_idx2]

            # create edge index
            num_vertex1, num_vertex2 = pos_3d_world1.shape[0], pos_3d_world2.shape[0]
            edges_idx1 = torch.tensor([[i, (i + 1) % num_vertex1] for i in range(num_vertex1)])
            edges_idx2 = torch.tensor([[i, (i + 1) % num_vertex2] for i in range(num_vertex2)])
            edges1, edges2 = pos_2d_1[edges_idx1], pos_2d_2[edges_idx2]

            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for edge in edges1:
                start = (edge[0].cpu().numpy() * img_ref_grey.shape[::1]).astype(np.int32)
                end = (edge[1].cpu().numpy() * img_ref_grey.shape[::1]).astype(np.int32)
                cv2.line(outImg, start, end, random_color, thickness=2)
            for edge in edges2:
                start = (edge[0].cpu().numpy() * img_src_grey.shape[::1]).astype(np.int32)
                end = (edge[1].cpu().numpy() * img_src_grey.shape[::1]).astype(np.int32)
                start += np.array([img_ref.shape[1], 0])
                end += np.array([img_ref.shape[1], 0])
                cv2.line(outImg, start, end, random_color, thickness=2)

            # draw sampled point
            sample_points_ref = sample_points[idx][0]
            sample_points_src = sample_points[idx][1]

            # project to 2d and draw
            sample_points_ref_c = (self.transformation_w2c_ref @ to_homogeneous_tensor(sample_points_ref).T).T
            sample_points_ref_c = sample_points_ref_c[:, :3] / (sample_points_ref_c[:, 3:4] + 1e-8)
            sample_points_2d_ref = (self.intrinsic @ sample_points_ref_c.T).T
            sample_points_2d_ref = sample_points_2d_ref[:, :2] / (sample_points_2d_ref[:, 2:3] + 1e-8)
            sample_points_2d_ref = (sample_points_2d_ref.cpu().numpy() * img_ref_grey.shape[::1]).astype(np.int32)
            sample_points_2d_ref = np.clip(sample_points_2d_ref, 0, img_ref_grey.shape[1] - 1)
            outImg[sample_points_2d_ref[:, 1], sample_points_2d_ref[:, 0]] = random_color

            sample_points_src_c = (self.transformation_w2c_src @ to_homogeneous_tensor(sample_points_src).T).T
            sample_points_src_c = sample_points_src_c[:, :3] / (sample_points_src_c[:, 3:4] + 1e-8)
            sample_points_2d_src = (self.intrinsic @ sample_points_src_c.T).T
            sample_points_2d_src = sample_points_2d_src[:, :2] / (sample_points_2d_src[:, 2:3] + 1e-8)
            sample_points_2d_src = (sample_points_2d_src.cpu().numpy() * img_src_grey.shape[::1]).astype(np.int32)
            sample_points_2d_src = np.clip(sample_points_2d_src, 0, img_src_grey.shape[1] - 1)
            outImg[sample_points_2d_src[:, 1], sample_points_2d_src[:, 0] + img_ref.shape[1]] = random_color

            # if they are matched, draw a connect line between them
            if patch1[0] == "N":
                continue
            patch_center1 = (patch_center1 * img_ref_grey.shape[::1]).astype(np.int32)
            patch_center2 = (patch_center2 * img_src_grey.shape[::1]).astype(np.int32)
            cv2.line(outImg, (patch_center1[0], patch_center1[1]),
                     (patch_center2[0] + img_ref.shape[1], patch_center2[1]),
                     random_color, thickness=2)

        if is_show_window:
            cv2.imshow(f'Matches {matches[0][0]} and {matches[0][1]}', outImg)
            cv2.waitKey(0)
        if is_save_file:
            cv2.imwrite(os.path.join(self.v_log_root, "matches_face.png"), outImg)
        return

    def save_wireframe(self, graph_src, graph_name, file_name=None, out_path=None, is_save_world_coord=True):
        graph = nx.convert_node_labels_to_integers(graph_src)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            file_path = out_path
        else:
            file_path = os.path.join(self.v_log_root, file_name)

        with open(file_path, 'w') as file:
            for node, data in graph.nodes(data=True):
                if data['graph_name'] == graph_name:
                    pos = data['pos_3d']
                else:
                    pos = data['pos_3d_tf']

                if is_save_world_coord:
                    pos = (self.transformation_c2w_ref @ to_homogeneous_tensor(pos.unsqueeze(0)).T).T
                    pos = pos[:, :3] / (pos[:, 3:4] + 1e-8)
                    pos = pos.squeeze(0)

                file.write(f'v {pos[0]} {pos[1]} {pos[2]}\n')
            for u, v in graph.edges():
                file.write(f'l {u + 1} {v + 1}\n')  # '+1' is because .obj file indexing starts from 1
        print("PairGraphMatcher: wireframe is save to {}".format(file_path))

    @staticmethod
    def save_point_cloud(point_cloud_np, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)
        o3d.io.write_point_cloud(file_path, point_cloud)
        print("PairGraphMatcher: point cloud is save to {}".format(file_path))


class PointSet:
    class Point:
        def __init__(self, _pos3d, _graph_name, _patch_id, _vertex_id):
            self.pos = _pos3d
            self.graph_name = _graph_name
            self.patch_id = _patch_id
            self.vertex_id = _vertex_id

    def __init__(self, pos3d, graph_name, patch_id, vertex_id):
        self.PointList: list[Point] = []
        self.pos3d = []
        self.add_point(pos3d, graph_name, patch_id, vertex_id)

    def add_point(self, pos3d, graph_name, patch_id, vertex_id):
        self.PointList.append(PointSet.Point(pos3d, graph_name, patch_id, vertex_id))
        self.pos3d.append(pos3d)

    def get_pos3d(self):
        return torch.mean(torch.stack(self.pos3d), dim=0)

    def extend_pointset(self, PointSet_add):
        self.PointList.extend(PointSet_add.PointList)
        self.pos3d.extend(PointSet_add.pos3d)

    def is_vaild(self, min_num=3):
        return len(self.PointList) >= min_num


class GraphFusion:
    def __init__(self, v_data, v_log_root, device):
        # src data
        self.v_img_database: list[Image] = v_data[0]
        self.v_graphs: np.ndarray[nx.Graph] = v_data[1]
        self.v_img_pairs: list[np.ndarray] = v_data[2]

        graph_loss = []
        for idx, graph in enumerate(self.v_graphs):
            graph.name = idx
            graph.graph['v_img'] = self.v_img_database[idx]
            graph.graph['v_img_pair'] = list(self.v_img_pairs[idx][:, 0].astype(np.int64))

            loss_list = []
            for patch_idx in range(len(graph.graph['dual_graph'].nodes)):
                loss_list.append(graph.graph['dual_graph'].nodes[patch_idx]['loss'])
            mean_loss = sum(loss_list) / len(loss_list)
            graph_loss.append(mean_loss)

        self.sort_idx = np.argsort(graph_loss).tolist()

        self.graphs = self.v_graphs
        # self.graphs = self.v_graphs[self.sort_idx[0:50]]

        # self.graphs = self.graphs.copy()  # self.graphs’ node are renamed and usd to create the match
        # Add a unique prefix to the nodes of each graph
        for i, G in enumerate(self.graphs):
            mapping = {node: f'G{G.name}_{node}' for node in G.nodes()}
            nx.relabel_nodes(G, mapping, copy=False)

        # self.graph_ref = self.graphs[0]
        # self.graphs_src = self.graphs[1::]
        # self.v_img_ref = self.graph_ref.graph["v_img"]
        # self.v_imgs_src = [graph_src.graph["v_img"] for graph_src in self.graphs_src]
        self.recon_graph = None
        self.super_graph = None
        self.super_graph_src = None
        self.CoarsedGraphMatcherManger: dict[tuple, PairGraphMatcher] = {}
        self.ExactedGraphMatcherManger: dict[tuple, PairGraphMatcher] = {}

        self.v_log_root = v_log_root
        self.v_log_root_plane_optimzing = os.path.join(os.path.dirname(self.v_log_root), "1.PlaneOptimizing")
        self.device = device

        # prepare the graph for fusion
        # self.graphs_pointset are usde to create super graph and fuse
        self.coarsed_matches_node = []
        self.coarsed_matches_face = []
        self.exacted_matches_node = []
        self.exacted_matches_face = []

        self.coarsed_group_node = None
        self.coarsed_group_face = None
        self.exacted_group_node = None
        self.exacted_group_face = None
        self.graphs_pointset = []
        self.merged_record = {}

    def run(self):
        print("GraphFusion: Begin to Pair Graph Matching")
        # exact match
        # 1. create the pairer graph matcher for each pair
        checked_match = set()
        for idx_ref, graph_ref in enumerate(self.graphs):
            idx_refs = graph_ref.graph['v_img_pair']
            # for idx2, graph_src in enumerate(self.graphs[idx1 + 1::]):
            #     if graph_ref.name == graph_src.name:
            #         continue
            # if graph_ref.name not in self.sort_idx[0:10]:
            #     continue
            for idx_ref in idx_refs:
                graph_src = self.graphs[idx_ref]
                # if not (graph_ref.name == 5 and graph_src.name == 13):
                #     continue
                if ((graph_ref.name, graph_src.name) in checked_match
                        or (graph_src.name, graph_ref.name) in checked_match):
                    continue
                checked_match.add((graph_ref.name, graph_src.name))

                v_log_root_c = os.path.join(self.v_log_root, str(graph_ref.name) + "_" + str(graph_src.name))
                # self.CoarsedGraphMatcherManger[(graph_ref.name, graph_src.name)] = PairGraphMatcher(graph_ref,
                #                                                                                     graph_src,
                #                                                                                     v_log_root_c,
                #                                                                                     self.device)
                self.ExactedGraphMatcherManger[(graph_ref.name, graph_src.name)] = PairGraphMatcher(graph_ref,
                                                                                                    graph_src,
                                                                                                    v_log_root_c,
                                                                                                    self.device,
                                                                                                    is_exact_match=True)

        # 2. run the pairer graph matcher to get the matches of nodes and faces
        for idx, gm in enumerate(list(self.CoarsedGraphMatcherManger.values())):
            print("Running CoarsedGraphMatcherManger: {}/{}".format(idx + 1, len(self.CoarsedGraphMatcherManger)))
            gm.run()
            self.coarsed_matches_node.extend(gm.recon_graph_matches_node)
            self.coarsed_matches_face.extend(gm.recon_graph_matches_face)

        for idx, gm in enumerate(list(self.ExactedGraphMatcherManger.values())):
            print("Running ExactedGraphMatcherManger: {}/{}".format(idx + 1, len(self.ExactedGraphMatcherManger)))
            gm.run()
            self.exacted_matches_node.extend(gm.recon_graph_matches_node)
            self.exacted_matches_face.extend(gm.recon_graph_matches_face)

        # 3. merge the matches of nodes and faces
        # self.coarsed_group_node = merge_matches(self.coarsed_matches_node)
        # self.coarsed_group_face = merge_matches(self.coarsed_matches_face)
        self.exacted_group_node = merge_matches(self.exacted_matches_node)
        self.exacted_group_face = merge_matches(self.exacted_matches_face)

        # 4. vis matched faces’ point cloud
        pc = []
        for gm in list(self.CoarsedGraphMatcherManger.values()):
            if gm.match_points:
                all_matched_pc = list(gm.match_points.values())
                pc.append(torch.cat(all_matched_pc))
                # if all_matched_pc[0].shape[0] < all_matched_pc[1].shape[0]:
                #     pc.append(all_matched_pc[0])
                # else:
                #     pc.append(all_matched_pc[1])

        if pc:
            pc = torch.cat(pc)
            PairGraphMatcher.save_point_cloud(pc.cpu().numpy(),
                                              os.path.join(self.v_log_root, "Coarsed_matched_point_cloud.ply"))

        pc = []
        for gm in list(self.ExactedGraphMatcherManger.values()):
            if gm.match_points:
                all_matched_pc = list(gm.match_points.values())
                pc.append(torch.cat(all_matched_pc))
                # if all_matched_pc[0].shape[0] < all_matched_pc[1].shape[0]:
                #     pc.append(all_matched_pc[0])
                # else:
                #     pc.append(all_matched_pc[1])
        if pc:
            pc = torch.cat(pc)
            PairGraphMatcher.save_point_cloud(pc.cpu().numpy(),
                                              os.path.join(self.v_log_root, "Exacted_matched_point_cloud.ply"))

        # self.vis_group_faces(self.group_face, self.matches_face, floder_name="group")
        # self.fuse_node(self.coarsed_matches_node)

        # for each face in the exacted match face group, we share their src_imgs
        id_src_imgs_group = {}
        for idx, group in enumerate(self.exacted_group_face):
            id_src_imgs_group_c = []
            for face in group:
                graph_id, face_id = int(face.split('_')[0][1::]), int(face.split('_')[1][1::])
                id_src_imgs = (self.v_img_pairs[graph_id][:, 0]).astype(np.int64)
                id_src_imgs_group_c.append(id_src_imgs)
            id_src_imgs_group_c = np.unique(np.concatenate(id_src_imgs_group_c))
            for face in group:
                id_src_imgs_group[face] = id_src_imgs_group_c

        # 6. reoptimize the face
        success_count = 0
        reoptimized_faces = []
        rechecked_matches = []
        for match in self.coarsed_matches_face:
            if match in self.exacted_matches_face or match[::-1] in self.exacted_matches_face:
                continue

            rechecked_matches.append(match)

            graph1_id, face1_id = int(match[0].split('_')[0][1::]), int(match[0].split('_')[1][1::])
            graph2_id, face2_id = int(match[1].split('_')[0][1::]), int(match[1].split('_')[1][1::])
            # TODO: continuing optimizing the face, but using more src_imgs
            id_src_imgs1 = (self.v_img_pairs[graph1_id][:, 0]).astype(np.int64)
            id_src_imgs2 = (self.v_img_pairs[graph2_id][:, 0]).astype(np.int64)
            id_src_imgs = np.unique(np.concatenate((id_src_imgs1, id_src_imgs2)))

            if match[0] in id_src_imgs_group:
                id_src_imgs = np.unique(np.concatenate((id_src_imgs, id_src_imgs_group[match[0]])))
            if match[1] in id_src_imgs_group:
                id_src_imgs = np.unique(np.concatenate((id_src_imgs, id_src_imgs_group[match[1]])))

            print("GraphFusion: Begin to reoptimize the face {} and {}, using {} source images".format(match[0],
                                                                                                       match[1],
                                                                                                       len(id_src_imgs)))
            src_Images = [self.v_img_database[id_src] for id_src in id_src_imgs]
            if match[0] not in reoptimized_faces:
                graph_optimiser1 = GraphPlaneOptimiser(graph1_id, self.v_graphs[graph1_id],
                                                       self.v_img_database[graph1_id],
                                                       src_Images,
                                                       os.path.join(self.v_log_root_plane_optimzing, str(graph1_id)),
                                                       self.device,
                                                       use_cache=True)

                graph_optimiser1.run(only_optimize_patch_id=[face1_id])
                reoptimized_faces.append(match[0])

            if match[1] not in reoptimized_faces:
                graph_optimiser2 = GraphPlaneOptimiser(graph2_id, self.v_graphs[graph2_id],
                                                       self.v_img_database[graph2_id],
                                                       src_Images,
                                                       os.path.join(self.v_log_root_plane_optimzing, str(graph2_id)),
                                                       self.device, use_cache=True)
                graph_optimiser2.run(only_optimize_patch_id=[face2_id])
                reoptimized_faces.append(match[1])

            if (graph1_id, graph2_id) in self.ExactedGraphMatcherManger:
                gm = self.ExactedGraphMatcherManger[(graph1_id, graph2_id)]
            else:
                gm = self.ExactedGraphMatcherManger[(graph2_id, graph1_id)]
            gm.run()
            new_exacted_matches_face = gm.recon_graph_matches_face

            if match in new_exacted_matches_face or match[::-1] in new_exacted_matches_face:
                success_count = success_count + 1
                print("Reoptimization Successful! {} and {} is exacted match".format(match[0], match[1]))
            else:
                print("Reoptimization Failed! {} and {} is not exacted match".format(match[0], match[1]))
        return

    def vis_group_faces(self, group_face, matches_face, floder_name="group"):
        # vis each group
        pc = []
        non_direct_matches = []
        for group in group_face:
            if len(group) < 2:
                continue

            pc_group_c = []
            non_direct_matches_c = []
            for idx1, graph_face1 in enumerate(group):
                for idx2, graph_face2 in enumerate(group[idx1 + 1::]):
                    graph1, face1 = int(graph_face1.split('_')[0][1::]), int(graph_face1.split('_')[1][1::])
                    graph2, face2 = int(graph_face2.split('_')[0][1::]), int(graph_face2.split('_')[1][1::])

                    if ((graph_face1, graph_face2) not in matches_face
                            and (graph_face2, graph_face1) not in matches_face):
                        non_direct_matches_c.append((graph_face1, graph_face2))
                        continue

                    if (graph1, graph2) in self.GraphMatcherManger:
                        pair_match = self.GraphMatcherManger[(graph1, graph2)]
                        pc_matched_face = pair_match.match_points[face1]
                    else:
                        pair_match = self.GraphMatcherManger[(graph2, graph1)]
                        pc_matched_face = pair_match.match_points[face2]
                    pc_group_c.append(pc_matched_face)

            non_direct_matches.append(non_direct_matches_c)
            pc.extend(pc_group_c)
            PairGraphMatcher.save_point_cloud(torch.cat(pc_group_c).cpu().numpy(),
                                              os.path.join(self.v_log_root, floder_name, "{}.ply".format(str(group))))

        PairGraphMatcher.save_point_cloud(torch.cat(pc).cpu().numpy(),
                                          os.path.join(self.v_log_root, floder_name,
                                                       "matched_point_cloud_filtered.ply"))
        return non_direct_matches

    def fuse_node(self, matches_node):
        # 5. fuse the graphs and use the matches to trim the graph
        print("GraphFusion: Begin to create pointset_graph for each graph")
        for idx, graph in enumerate(self.graphs):
            c_graph = self.create_graph(graph)
            if c_graph is not None:
                self.graphs_pointset.append(c_graph)

        # SuperGraph
        print("GraphFusion: Begin to create super graph")
        node_num = sum([len(graph.nodes) for graph in self.graphs_pointset])
        edge_num = sum([len(graph.edges) for graph in self.graphs_pointset])
        self.super_graph = nx.compose_all(self.graphs_pointset)
        self.super_graph_src = self.super_graph.copy()

        assert len(self.super_graph.nodes) == node_num and len(self.super_graph.edges) == edge_num
        self.save_wireframe(self.super_graph, "super_graph.obj")

        print("GraphFusion: Begin to Fusing")
        # merge the matched nodes in supergraph
        for n1, n2 in matches_node:
            self.merge_node(self.super_graph, n1, n2)

        self.save_wireframe(self.super_graph, "super_graph_fused.obj", is_find_faces=False)

        matches_node_set = set()
        for n1, n2 in matches_node:
            matches_node_set.add(n1)
            matches_node_set.add(n2)

        # 1. del node
        nodes_to_remove = []
        for node in self.super_graph.nodes:
            data = self.super_graph.nodes[node]['data']
            # if not data.is_vaild():
            if node not in matches_node_set:
                nodes_to_remove.append(node)
        self.super_graph.remove_nodes_from(nodes_to_remove)
        self.save_wireframe(self.super_graph, "super_graph_fused_trim_node.obj")

        # 2. del edge
        # edges_to_remove = [(u, v) for u, v, d in self.super_graph.edges(data=True)
        #                    if 'weight' not in d or d['weight'] < 1]
        # self.super_graph.remove_edges_from(edges_to_remove)
        # self.save_wireframe(self.super_graph, "super_graph_fused_trim_node3_edge1.obj")

        print("GraphFusion: End")

    def create_graph(self, graph):
        print("Creating graph {}".format(graph.name))

        # prepare some data
        patches = [graph.graph["dual_graph"].nodes[i]["id_vertex"] for i in graph.graph["dual_graph"].nodes]
        rays = torch.from_numpy(np.stack([graph.nodes[i]["ray_c"] for i in graph.nodes])).to(self.device).to(
                torch.float32)
        optimized_abcd_list = graph.graph["dual_graph"].graph['optimized_abcd_list']
        intrinsic = torch.from_numpy(graph.graph["v_img"].intrinsic).to(self.device).to(torch.float32)

        recon_graph = nx.Graph()
        recon_graph.name = graph.name
        recon_graph.graph["v_img"] = graph.graph["v_img"]

        # checking the vaildity of the patches
        patch_loss = [graph.graph['dual_graph'].nodes[i]['loss'] for i in graph.graph['dual_graph'].nodes]
        mean_patch_loss = sum(patch_loss) / len(patch_loss)
        isvalid_patch = [loss < 0.5 for loss in patch_loss]
        # isvalid_patch = [True for loss in patch_loss]
        if not sum(isvalid_patch):
            print("\033[91mError: Checking failed! No valid patches in graph {}\033[0m".format(graph.name))
            return None
        elif sum(isvalid_patch) < len(isvalid_patch):
            print("\033[93mWarning: Filtering {} invalid patches in graph {}\033[0m".format(
                    len(isvalid_patch) - sum(isvalid_patch),
                    graph.name))

        for patch_idx, patch_c in enumerate(patches):
            if not isvalid_patch[patch_idx]:
                continue

            # add nodes, the names of nodes are vertex_id of graph
            pos_3d_c = intersection_of_ray_and_all_plane(optimized_abcd_list[patch_idx].unsqueeze(0), rays[patch_c])[0]
            transformation_w2c = torch.from_numpy(graph.graph["v_img"].extrinsic).to(self.device).to(torch.float32)
            pos_3d_w = (torch.linalg.inv(transformation_w2c) @ to_homogeneous_tensor(pos_3d_c).T).T
            pos_3d_w = pos_3d_w[:, :3] / (pos_3d_w[:, 3:4] + 1e-8)

            pos_2d = (intrinsic @ pos_3d_c.T).T
            pos_2d = pos_2d[:, :2] / (pos_2d[:, 2:3] + 1e-8)
            pos_2d_src = np.array([graph.nodes[f'G{graph.name}_{vertex_id}']["pos_2d"] for vertex_id in patch_c],
                                  dtype=np.float32)
            pos_2d_src = torch.from_numpy(pos_2d_src).to(self.device).to(torch.float32)

            for idx, (pos_3d_w_, pos_2d_) in enumerate(zip(pos_3d_w, pos_2d)):
                recon_graph.add_node(f'G{graph.name}_{patch_c[idx]}',
                                     data=PointSet(pos_3d_w_, graph.name, patch_idx, patch_c[idx]))

            id_edge_per_patch = [(f'G{graph.name}_{patch_c[idx - 1]}', f'G{graph.name}_{patch_c[idx]}')
                                 for idx in range(len(patch_c))]
            recon_graph.add_edges_from(id_edge_per_patch)
        return recon_graph

    # merge n2 to n1
    def merge_node(self, graph, n1, n2):
        while n1 in self.merged_record:
            n1 = self.merged_record[n1]
        while n2 in self.merged_record:
            n2 = self.merged_record[n2]

        if n1 == n2:
            return

        assert n1 in graph.nodes and n2 in graph.nodes
        graph.nodes[n1]['data'].extend_pointset(graph.nodes[n2]['data'])
        for neighbor_of_n2 in graph.neighbors(n2):
            if neighbor_of_n2 == n1:
                # shouldn't happen, matches nodes are not neighbors
                continue
            if (n1, neighbor_of_n2) not in graph.edges:
                graph.add_edge(n1, neighbor_of_n2)

        graph.remove_node(n2)
        self.merged_record[n2] = n1

    def add(self):
        pass

    def save_wireframe(self, graph_src, file_name, is_find_faces=False):
        graph = nx.convert_node_labels_to_integers(graph_src)
        file_path = os.path.join(self.v_log_root, file_name)
        faces = None
        if is_find_faces:
            faces = nx.cycle_basis(graph)

        with open(file_path, 'w') as file:
            for node, data in graph.nodes(data=True):
                pointset = data['data']
                pos = pointset.get_pos3d()
                file.write(f'v {pos[0]} {pos[1]} {pos[2]}\n')

            for u, v in graph.edges():
                file.write(f'l {u + 1} {v + 1}\n')  # '+1' is because .obj file indexing starts from 1

            if faces:
                for face in faces:
                    face_indices = [str(node + 1) for node in face]
                    file.write(f'f {" ".join(face_indices)}\n')

        print("GraphFusion: super graph is save to {}".format(file_path))

    def reoptimization(self):
        def find_no_direct_match(group, matches):
            not_direct_matches = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if (group[i], group[j]) not in matches or (group[j], group[i]) not in matches:
                        not_direct_matches.append((group[i], group[j]))
            return not_direct_matches

        for group_c in self.group_face:
            print("Group {}:".format(group_c))
            not_direct_matches_c = find_no_direct_match(group_c, self.matches_face)
            if not not_direct_matches_c:
                continue

            print("Begin Reoptimization: Not direct matches {}".format(not_direct_matches_c))

        pass
