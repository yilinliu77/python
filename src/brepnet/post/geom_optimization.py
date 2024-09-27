import os
import torch
import torch.nn as nn
import numpy as np
from chamferdist import ChamferDistance
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import trimesh


def get_bbox_minmax(point_cloud):
    min_point, _ = torch.min(point_cloud.reshape(-1, 3), dim=0)
    max_point, _ = torch.max(point_cloud.reshape(-1, 3), dim=0)
    return min_point, max_point


def compute_bbox_center_and_size(min_corner, max_corner):
    center = (min_corner + max_corner) / 2
    size = torch.max(max_corner - min_corner)
    return center, size


def get_size_and_center(point_cloud):
    obj_sc = []
    for idx in range(point_cloud.shape[0]):
        min_point, max_point = get_bbox_minmax(point_cloud[idx])
        center, size = compute_bbox_center_and_size(min_point, max_point)
        obj_sc.append(torch.tensor([size, center[0], center[1], center[2]]))
    return torch.stack(obj_sc, dim=0).to(point_cloud.device)


def apply_transform(tensor, transform):
    src_shape = tensor.shape
    if len(src_shape) > 2:
        tensor = tensor.reshape(-1, 3)
    scale, offset = transform[0:1], transform[1:]
    center = torch.mean(tensor, dim=0)
    scaled_tensor = (tensor - center) * scale + center + offset
    if len(src_shape) > 2:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def apply_transform_batch(tensor, transform):
    src_shape = tensor.shape
    if len(src_shape) > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, 3)
    scales = transform[:, 0].view(-1, 1, 1)
    offsets = transform[:, 1:].view(-1, 1, 3)
    centers = tensor.mean(dim=1, keepdim=True)
    scaled_tensor = (tensor - centers) * scales + centers + offsets
    if len(src_shape) > 3:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def get_wire_length(face_edge):
    return torch.norm(face_edge[:, 1::, :] - face_edge[:, 0:-1, :], dim=-1).sum()


def interpolate_edge_points(edge_points, num_interpolations):
    interpolated_points = []
    for i in range(len(edge_points) - 1):
        start_point = edge_points[i]
        end_point = edge_points[i + 1]
        for t in np.linspace(0, 1, num_interpolations + 2):  # +2 to include endpoints
            interpolated_point = (1 - t) * start_point + t * end_point
            interpolated_points.append(interpolated_point)
    return np.array(interpolated_points)


def check_edge_validity(edge_point, face_point1, face_point2, tol=1e-2):
    edge_point = interpolate_edge_points(edge_point, 10)

    computer = ChamferDistance()
    edge_point = torch.from_numpy(edge_point)
    face_point1 = torch.from_numpy(face_point1)
    face_point2 = torch.from_numpy(face_point2)
    dst1 = computer(edge_point.reshape(1, -1, 3), face_point1.reshape(1, -1, 3), point_reduction='mean')
    dst2 = computer(edge_point.reshape(1, -1, 3), face_point2.reshape(1, -1, 3), point_reduction='mean')
    dis = (dst1 + dst2) / 2

    if dst1 < tol and dst2 < tol:
        return True, dis
    else:
        return False, dis

    # return (abs(dst1 - dst2) / torch.min(dst1, dst2)) < 0.25


def fix_init_estimation(recon_face, recon_edge, edge_face_connectivity, face_edge_adj):
    # 1. Check and init fix the connectivity of the edge and face
    cache_dict = {}
    for conec in edge_face_connectivity:
        if (conec[1], conec[2]) in cache_dict:
            cache_dict[(conec[1], conec[2])].append(conec[0])
        elif (conec[2], conec[1]) in cache_dict:
            cache_dict[(conec[2], conec[1])].append(conec[0])
        else:
            cache_dict[(conec[1], conec[2])] = [conec[0]]

    # Duplicate or delete edges if it appears once
    new_dict = {}
    del_key = []
    for key, value in cache_dict.items():
        if len(value) == 1:
            is_egde_valid, dis = check_edge_validity(recon_edge[value[0]], recon_face[key[0]], recon_face[key[1]])
            if is_egde_valid:
                # should fix the missing edge
                new_edge = recon_edge[value[0]][::-1][None, :]
                edge_face_connectivity = np.concatenate([
                    edge_face_connectivity,
                    np.array((recon_edge.shape[0], key[1], key[0]))[None, :]
                ], axis=0)
                new_dict[key] = recon_edge.shape[0]
                face_edge_adj[key[1]].append(recon_edge.shape[0])
                recon_edge = np.concatenate([recon_edge, new_edge], axis=0)
            else:
                # remove the invalid edge
                edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 0] != value[0]]
                if value[0] in face_edge_adj[key[0]]:
                    face_edge_adj[key[0]].remove(value[0])
                elif value[0] in face_edge_adj[key[1]]:
                    face_edge_adj[key[1]].remove(value[0])
                del_key.append(key)

    for key, value in new_dict.items():
        cache_dict[key].append(value)
    for key in del_key:
        del cache_dict[key]

    # 2. Check which half edge is better and replace the bad one
    computer = ChamferDistance()
    for (id_face1, id_face2), id_edges in cache_dict.items():
        face1 = torch.from_numpy(recon_face[id_face1])
        face2 = torch.from_numpy(recon_face[id_face2])
        edge1 = torch.from_numpy(recon_edge[id_edges[0]])
        edge2 = torch.from_numpy(recon_edge[id_edges[1]])
        dist1 = (computer(edge1.reshape(1, -1, 3), face1.reshape(1, -1, 3)) +
                 computer(edge1.reshape(1, -1, 3), face2.reshape(1, -1, 3)))
        dist2 = (computer(edge2.reshape(1, -1, 3), face1.reshape(1, -1, 3)) +
                 computer(edge2.reshape(1, -1, 3), face2.reshape(1, -1, 3)))
        if dist1 > dist2:
            recon_edge[id_edges[0]] = recon_edge[id_edges[1]][::-1]
        elif dist1 < dist2:
            recon_edge[id_edges[1]] = recon_edge[id_edges[0]][::-1]
    return recon_face, recon_edge, edge_face_connectivity, face_edge_adj


def fix_final_estimation(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, tol=2e-2):
    # check and remove the invalid edge
    remove_edge_idx = []
    for edge_idx, face_idx1, face_idx2 in edge_face_connectivity:
        is_egde_valid, dis = check_edge_validity(recon_edge[edge_idx], recon_face[face_idx1], recon_face[face_idx2], tol=tol)
        if not is_egde_valid:
            edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 0] == edge_idx]
            if edge_idx in face_edge_adj[face_idx1]:
                face_edge_adj[face_idx1].remove(edge_idx)
            elif edge_idx in face_edge_adj[face_idx2]:
                face_edge_adj[face_idx2].remove(edge_idx)
            remove_edge_idx.append(edge_idx)
    return recon_face, recon_edge, edge_face_connectivity, face_edge_adj, remove_edge_idx


def merge_edge_endpoints(edge_wcs, tol):
    points = edge_wcs[:, [0, -1]].reshape(-1, 3)  # N*2

    merged_points = []
    visited = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if visited[i]:
            continue
        distances = np.linalg.norm(points - point, axis=1)
        close_points = points[distances <= tol]
        merged_point = close_points.mean(axis=0)
        merged_points.append(merged_point)
        visited[distances <= tol] = True
    merged_points = np.array(merged_points)

    num_point = edge_wcs.shape[1]
    for i in range(len(edge_wcs)):
        edge_wcs[i][0] = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][0], axis=1))]
        edge_wcs[i][-1] = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][-1], axis=1))]
        # start_vec = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][0], axis=1))] - edge_wcs[i][0]
        # end_vec = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][-1], axis=1))] - edge_wcs[i][-1]
        # weight = np.tile((np.arange(num_point) / (num_point - 1))[:, np.newaxis], (1, 3))
        # weighted_vec = np.tile(start_vec[np.newaxis, :], (num_point, 1)) * (1 - weight) + np.tile(end_vec, (num_point, 1)) * weight
        # edge_wcs[i] += weighted_vec

    return edge_wcs


class STModel(nn.Module):
    def __init__(self, init_surf_st, init_edge_st):
        super().__init__()
        self.surf_st = torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(init_surf_st, 1)
        # self.surf_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(init_surf_st, 1))
        self.edge_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(init_edge_st, 1))


class Geom_Optimization():
    def __init__(self, recon_face, recon_edge, edge_face_connectivity, face_edge_adj, min_vertex_tolerance=5e-3, max_vertex_tolerance=0.25,
                 max_iter=100):
        self.edge_face_connectivity = edge_face_connectivity
        self.face_edge_adj = face_edge_adj

        self.device = torch.device("cpu")
        # self.device = torch.device("cuda")

        self.recon_face = torch.FloatTensor(recon_face).to(self.device).requires_grad_(False)
        res = self.recon_face.shape[1]
        densities = (torch.linalg.norm(
            self.recon_face[:,0,0] - self.recon_face[:,res//2,res//2], dim=-1) * 10).to(torch.long) * res

        self.face_points_candidates = []
        for face,density in zip(self.recon_face, densities):
            x = torch.linspace(-1, 1, density).to(self.device)
            y = torch.linspace(-1, 1, density).to(self.device)
            x, y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
            self.face_points_candidates.append(torch.nn.functional.grid_sample(
                face[None].permute(0,3,1,2), coords, align_corners=True)[0,:,:,0].permute(1,0))
        self.recon_edge = torch.FloatTensor(recon_edge).to(self.device).requires_grad_(False)

        self.transformed_recon_face = torch.FloatTensor(recon_face).to(self.device).requires_grad_(False)
        self.transformed_recon_edge = torch.FloatTensor(recon_edge).to(self.device).requires_grad_(False)

        inv_edge_face_connectivity = {}
        for edge_idx, face_idx1, face_idx2 in edge_face_connectivity:
            inv_edge_face_connectivity[(face_idx1, face_idx2)] = edge_idx

        pair1 = []
        pair2 = []
        face_adj = np.zeros((recon_face.shape[0], recon_face.shape[0]), dtype=bool)
        face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True
        # Set the diagonal to False
        np.fill_diagonal(face_adj, False)
        face_adj = np.logical_or(face_adj, face_adj.T)
        # Find the 3-node ring of the face
        for face1 in range(self.recon_face.shape[0]):
            for face2 in range(self.recon_face.shape[0]):
                if not face_adj[face1, face2]:
                    continue
                for face3 in range(self.recon_face.shape[0]):
                    if not face_adj[face1, face3]:
                        continue
                    if not face_adj[face2, face3]:
                        continue

                    g1 = [inv_edge_face_connectivity[(face1, face2)]]

                    def dis(a, b):
                        return torch.norm(self.recon_edge[a, 0] - self.recon_edge[b, 0])
                    if (dis(g1[0], inv_edge_face_connectivity[(face1, face3)])
                            < dis(g1[0], inv_edge_face_connectivity[(face3, face1)])):
                        g1.append(inv_edge_face_connectivity[(face1, face3)])
                    else:
                        g1.append(inv_edge_face_connectivity[(face3, face1)])

                    if (dis(g1[0], inv_edge_face_connectivity[(face2, face3)])
                            < dis(g1[0], inv_edge_face_connectivity[(face3, face2)])):
                        g1.append(inv_edge_face_connectivity[(face2, face3)])
                    else:
                        g1.append(inv_edge_face_connectivity[(face3, face2)])

                    g2 = [inv_edge_face_connectivity[(face2, face1)]]
                    if (dis(g2[0], inv_edge_face_connectivity[(face2, face3)])
                            < dis(g2[0], inv_edge_face_connectivity[(face3, face2)])):
                        g2.append(inv_edge_face_connectivity[(face2, face3)])
                    else:
                        g2.append(inv_edge_face_connectivity[(face3, face2)])

                    if (dis(g2[0], inv_edge_face_connectivity[(face1, face3)])
                            < dis(g2[0], inv_edge_face_connectivity[(face3, face1)])):
                        g2.append(inv_edge_face_connectivity[(face1, face3)])
                    else:
                        g2.append(inv_edge_face_connectivity[(face3, face1)])

                    dis1 = self.recon_edge[g1]
                    dis1 = torch.norm(dis1[0, 0] - dis1[1, 0], dim=-1) + torch.norm(dis1[0, 0] - dis1[2, 0], dim=-1) + torch.norm(dis1[1, 0] - dis1[2, 0], dim=-1)

                    dis2 = self.recon_edge[g2]
                    dis2 = torch.norm(dis2[0, 0] - dis2[1, 0], dim=-1) + torch.norm(dis2[0, 0] - dis2[2, 0], dim=-1) + torch.norm(dis2[1, 0] - dis2[2, 0], dim=-1)

                    if dis1 > 3e-1 and dis2 > 3e-1:
                        continue

                    if dis1.sum() < dis2.sum():
                        pair1.append(g1)
                    else:
                        pair1.append(g2)

                    g2 = []
                    if inv_edge_face_connectivity[(face1, face2)] in pair1[-1]:
                        g2.append(inv_edge_face_connectivity[(face2, face1)])
                    else:
                        g2.append(inv_edge_face_connectivity[(face1, face2)])
                    if inv_edge_face_connectivity[(face1, face3)] in pair1[-1]:
                        g2.append(inv_edge_face_connectivity[(face3, face1)])
                    else:
                        g2.append(inv_edge_face_connectivity[(face1, face3)])
                    if inv_edge_face_connectivity[(face2, face3)] in pair1[-1]:
                        g2.append(inv_edge_face_connectivity[(face3, face2)])
                    else:
                        g2.append(inv_edge_face_connectivity[(face2, face3)])
                    pair2.append(g2)

        self.pair1 = np.asarray(pair1)
        self.pair2 = np.asarray(pair2)
        vertexes = self.recon_edge[:, [0, -1], :].reshape(-1, 3)
        dist_matrix = torch.sqrt(torch.clamp(torch.sum((vertexes.unsqueeze(1) - vertexes.unsqueeze(0)) ** 2, dim=-1), min=1e-6))
        self.penalized_vertex = torch.stack(
                torch.where(torch.logical_and(min_vertex_tolerance < dist_matrix, dist_matrix < max_vertex_tolerance)), dim=1)

        # init_surf_st, init_edge_st = self.get_init_st()
        self.model = STModel(self.recon_face.shape[0], self.recon_edge.shape[0])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.model = self.model.to(self.device).train()
        self.model.surf_st = self.model.surf_st.to(self.device)

        self.chamfer_dist = ChamferDistance()
        self.max_iter = max_iter

    def apply_all_transform(self):
        # for face_idx in range(self.recon_face.shape[0]):
        #     self.transformed_recon_face[face_idx] = apply_transform(self.recon_face[face_idx], self.model.surf_st[face_idx])
        # for edge_idx in range(self.recon_edge.shape[0]):
        #     self.transformed_recon_edge[edge_idx] = apply_transform(self.recon_edge[edge_idx], self.model.edge_st[edge_idx])
        # self.save1 = self.transformed_recon_face.clone()
        # self.save2 = self.transformed_recon_edge.clone()

        # self.transformed_recon_face = apply_transform_batch(self.recon_face, self.model.surf_st)
        self.transformed_recon_edge = apply_transform_batch(self.recon_edge, self.model.edge_st)

        # assert torch.allclose(self.save1, self.transformed_recon_face)
        # assert torch.allclose(self.save2, self.transformed_recon_edge)

    def dist(self, pc1, pc2, point_reduction='mean'):
        return self.chamfer_dist(
            pc1.reshape(1, -1, 3), pc2.reshape(1, -1, 3), batch_reduction="mean", point_reduction=point_reduction)

    def get_optimized_mask(self, tol=5e-3):
        need_optimized = []
        for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
            edge_to_face1 = self.dist(self.transformed_recon_edge[edge_idx], self.transformed_recon_face[face_idx1])
            edge_to_face2 = self.dist(self.transformed_recon_edge[edge_idx], self.transformed_recon_face[face_idx2])
            adj_distance_loss_c = (edge_to_face1 + edge_to_face2) / 2
            if adj_distance_loss_c > tol:
                need_optimized.append(edge_idx)

    def loss(self):
        adj_distance_loss = 0
        for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
            edge_to_face1 = self.dist(self.transformed_recon_edge[edge_idx], self.face_points_candidates[face_idx1])
            edge_to_face2 = self.dist(self.transformed_recon_edge[edge_idx], self.face_points_candidates[face_idx2])

            # edge_to_face1 = torch.cdist(self.transformed_recon_edge[edge_idx],
            #             self.transformed_recon_face[face_idx1].reshape(-1, 3)).min(dim=1)[0].mean()
            # edge_to_face2 = torch.cdist(self.transformed_recon_edge[edge_idx],
            #                             self.transformed_recon_face[face_idx2].reshape(-1, 3)).min(dim=1)[0].mean()

            adj_distance_loss_c = (edge_to_face1 + edge_to_face2) / 2
            # if adj_distance_loss_c > 5e-3:
            adj_distance_loss += adj_distance_loss_c

        endpoints = self.transformed_recon_edge[self.pair1, 0]
        endpoints_loss1 = torch.linalg.norm(endpoints[:,0] - endpoints[:,1], dim=-1).mean()
        endpoints_loss2 = torch.linalg.norm(endpoints[:,0] - endpoints[:,2], dim=-1).mean()
        endpoints_loss3 = torch.linalg.norm(endpoints[:,1] - endpoints[:,2], dim=-1).mean()

        endpoints = self.transformed_recon_edge[self.pair2, -1]
        endpoints_loss4 = torch.linalg.norm(endpoints[:,0] - endpoints[:,1], dim=-1).mean()
        endpoints_loss5 = torch.linalg.norm(endpoints[:,0] - endpoints[:,2], dim=-1).mean()
        endpoints_loss6 = torch.linalg.norm(endpoints[:,1] - endpoints[:,2], dim=-1).mean()
        endpoints_loss = (endpoints_loss1 + endpoints_loss2 + endpoints_loss3 + endpoints_loss4 + endpoints_loss5 + endpoints_loss6) / 6

        if self.penalized_vertex.shape[0] != 0:
            penalized_vertex_pair = self.transformed_recon_edge[:, [0, -1], :].reshape(-1, 3)[self.penalized_vertex]
            compat_vertexes_loss = torch.norm(penalized_vertex_pair[:, 0, :] - penalized_vertex_pair[:, 1, :], dim=-1).mean()
        else:
            compat_vertexes_loss = 0

        # sum_loss = adj_distance_loss + compat_vertexes_loss
        sum_loss = adj_distance_loss + endpoints_loss
        # sum_loss = endpoints_loss
        return sum_loss, adj_distance_loss.detach(), endpoints_loss.detach()

    def run(self):
        with tqdm(total=self.max_iter, desc='Geom Optimization', unit='iter') as pbar:
            for iter in range(self.max_iter):
                self.apply_all_transform()
                loss, adj_distance_loss, endpoints_loss = self.loss()
                self.optimizer.zero_grad()
                if loss < 0.001:
                    print(f'Early stop at iter {iter}')
                    break
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(
                    loss=loss.item(), adj=adj_distance_loss.cpu().item(), end=endpoints_loss.cpu().item())
                pbar.update(1)
        print('Optimization finished!')

    def get_transformed_data(self):
        return (self.transformed_recon_face.detach().cpu().numpy(),
                self.transformed_recon_edge.detach().cpu().numpy(),
                self.edge_face_connectivity,
                self.face_edge_adj
                )

    def get_transform(self):
        return self.model.surf_st.detach().cpu().numpy(), self.model.edge_st.detach().cpu().numpy()


def optimize_geom(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, max_iter=100):
    # 1. Check and init fix the connectivity of the edge and face
    # 2. Check which half edge is better and replace the bad one
    recon_face, recon_edge, edge_face_connectivity, face_edge_adj = fix_init_estimation(recon_face, recon_edge,
                                                                                        edge_face_connectivity,
                                                                                        face_edge_adj)
    # 3. Optimize the geometry to enclose the adjacent face and edge
    geom_opt = Geom_Optimization(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, max_iter=max_iter)
    geom_opt.run()
    recon_face, recon_edge, edge_face_connectivity, face_edge_adj = geom_opt.get_transformed_data()

    dirs = (recon_edge[:,[0,-1]]-np.mean(recon_edge, axis=1,keepdims=True))
    cos_dir = (dirs[:,0]*dirs[:,1]).sum(axis=1)/np.linalg.norm(dirs[:,0],axis=1)/np.linalg.norm(dirs[:,1],axis=1)
    flag = cos_dir > 0.75

    delta = recon_edge[:, [0,-1]].mean(axis=1)
    recon_edge[flag,0] = delta[flag]
    recon_edge[flag,-1] = delta[flag]

    # 4. Force to merge the edge endpoints
    # recon_edge = merge_edge_endpoints(recon_edge, 0.03)

    # 5. Check and remove the invalid edge
    remove_invalid_edge = True
    if remove_invalid_edge:
        recon_face, recon_edge, edge_face_connectivity, face_edge_adj, remove_edge_idx = fix_final_estimation(recon_face, recon_edge,
                                                                                                              edge_face_connectivity,
                                                                                                              face_edge_adj)

        return recon_face, recon_edge, edge_face_connectivity, face_edge_adj, remove_edge_idx
    else:
        return recon_face, recon_edge, edge_face_connectivity, face_edge_adj, []


def test_optimize_geom(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, debug_face_save_path):
    if not os.path.exists(debug_face_save_path):
        os.makedirs(debug_face_save_path, exist_ok=True)
    from shared.common_utils import export_point_cloud
    export_point_cloud(os.path.join(debug_face_save_path, 'src_face.ply'), recon_face.reshape(-1, 3))
    export_point_cloud(os.path.join(debug_face_save_path, 'src_edge.ply'), recon_edge.reshape(-1, 3))

    # add some offset to the points
    noised_face = recon_face + np.random.normal(0, 1e-2, size=(recon_face.shape[0], 1, 1, 1))
    noised_edge = recon_edge + np.random.normal(0, 1e-2, size=(recon_edge.shape[0], 1, 1))
    print(f"face mse after add noise: {np.mean((noised_face - recon_face) ** 2)}")
    print(f"line mse after add noise: {np.mean((noised_edge - recon_edge) ** 2)}")
    export_point_cloud(os.path.join(debug_face_save_path, 'noised_face.ply'), noised_face.reshape(-1, 3))
    export_point_cloud(os.path.join(debug_face_save_path, 'noised_edge.ply'), noised_edge.reshape(-1, 3))

    optimized_recon_faces, optimized_recon_edges, _, _ = optimize_geom(noised_face, noised_edge, edge_face_connectivity, face_edge_adj)

    export_point_cloud(os.path.join(debug_face_save_path, 'optimized_face.ply'), optimized_recon_faces.reshape(-1, 3))
    export_point_cloud(os.path.join(debug_face_save_path, 'optimized_edge.ply'), optimized_recon_edges.reshape(-1, 3))
    print(f"face mse after optimization: {np.mean((optimized_recon_faces - recon_face) ** 2)}")
    print(f"line mse after optimization: {np.mean((optimized_recon_edges - recon_edge) ** 2)}")
    return


if __name__ == '__main__':
    recon_face = np.random.rand(10, 16, 16, 3)
    recon_edge = np.random.rand(10, 16, 3)
    edge_face_connectivity = np.random.randint(0, 10, (10, 3))
    face_edge_adj = np.random.randint(0, 10, (10, 3))
    updated_face, updated_edge, _, _ = optimize_geom(recon_face, recon_edge, edge_face_connectivity, face_edge_adj)
