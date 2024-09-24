import os
import torch
import torch.nn as nn
import numpy as np
from chamferdist import ChamferDistance
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


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
    scale, offset = transform[0], transform[1:]
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


def check_edge_validity(edge_point, face_point1, face_point2):
    computer = ChamferDistance()
    edge_point = torch.from_numpy(edge_point)
    face_point1 = torch.from_numpy(face_point1)
    face_point2 = torch.from_numpy(face_point2)
    dst1 = computer(edge_point.reshape(1, -1, 3), face_point1.reshape(1, -1, 3))
    dst2 = computer(edge_point.reshape(1, -1, 3), face_point2.reshape(1, -1, 3))
    if dst1 < 5e-3 and dst2 < 5e-3:
        return True
    else:
        return False
    # return (abs(dst1 - dst2) / torch.min(dst1, dst2)) < 0.25


class STModel(nn.Module):
    def __init__(self, init_surf_st, init_edge_st):
        super().__init__()
        self.surf_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(init_surf_st, 1))
        self.edge_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(init_edge_st, 1))


class Geom_Optimization():
    def __init__(self, recon_face, recon_edge, edge_face_connectivity, face_edge_adj, vertex_tolerance=1e-1, max_iter=100):
        self.edge_face_connectivity = edge_face_connectivity
        self.face_edge_adj = face_edge_adj
        recon_face, recon_edge = self.get_init_estimation(recon_face, recon_edge)

        self.recon_face = torch.FloatTensor(recon_face).cuda().requires_grad_(False)
        self.recon_edge = torch.FloatTensor(recon_edge).cuda().requires_grad_(False)

        self.transformed_recon_face = torch.FloatTensor(recon_face).cuda().requires_grad_(False)
        self.transformed_recon_edge = torch.FloatTensor(recon_edge).cuda().requires_grad_(False)

        self.edge_face_connectivity = edge_face_connectivity
        self.face_edge_adj = face_edge_adj

        # self.face_idx_pair_to_edge_idx = {}
        # for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
        #     face_idx_pair = frozenset([face_idx1, face_idx2])
        #     if face_idx_pair not in self.face_idx_pair_to_edge_idx:
        #         self.face_idx_pair_to_edge_idx[face_idx_pair] = []
        #     self.face_idx_pair_to_edge_idx[face_idx_pair].append(edge_idx)
        # self.coincident_edge_idx = list(self.face_idx_pair_to_edge_idx.values())
        # self.coincident_edge_idx = torch.tensor(self.coincident_edge_idx)

        self.vertex_tol = vertex_tolerance
        vertexes = self.recon_edge[:, [0, -1], :].reshape(-1, 3)
        dist_matrix = torch.sqrt(torch.clamp(torch.sum((vertexes.unsqueeze(1) - vertexes.unsqueeze(0)) ** 2, dim=-1), min=1e-6))
        self.penalized_vertex = torch.stack(
                torch.where(torch.logical_and(self.vertex_tol * 0.01 < dist_matrix, dist_matrix < self.vertex_tol)), dim=1)

        # init_surf_st, init_edge_st = self.get_init_st()
        self.model = STModel(self.recon_face.shape[0], self.recon_edge.shape[0])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08, )
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)

        self.model = self.model.cuda().train()

        self.chamfer_dist = ChamferDistance()
        self.max_iter = max_iter

    def get_init_estimation(self, recon_face, recon_edge):
        edge_face_connectivity = self.edge_face_connectivity
        face_edge_adj = self.face_edge_adj
        cache_dict = {}
        for conec in edge_face_connectivity:
            if (conec[1], conec[2]) in cache_dict:
                cache_dict[(conec[1], conec[2])].append(conec[0])
            elif (conec[2], conec[1]) in cache_dict:
                cache_dict[(conec[2], conec[1])].append(conec[0])
            else:
                cache_dict[(conec[1], conec[2])] = [conec[0]]

        # Duplicate edges if it appears once
        new_dict = {}
        del_key = []
        for key, value in cache_dict.items():
            if len(value) == 1:
                is_egde_valid = check_edge_validity(recon_edge[value[0]], recon_face[key[0]], recon_face[key[1]])
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
            # else:
            #     is_egde_valid = check_edge_validity(recon_edge[value[0]], recon_face[key[0]], recon_face[key[1]])
            #     assert is_egde_valid

        for key, value in new_dict.items():
            cache_dict[key].append(value)
        for key in del_key:
            del cache_dict[key]

        # Check which half edge is better
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
        return recon_face, recon_edge

    def apply_all_transform(self):
        # for face_idx in range(self.recon_face.shape[0]):
        #     self.transformed_recon_face[face_idx] = apply_transform(self.recon_face[face_idx], self.model.surf_st[face_idx])
        # for edge_idx in range(self.recon_edge.shape[0]):
        #     self.transformed_recon_edge[edge_idx] = apply_transform(self.recon_edge[edge_idx], self.model.edge_st[edge_idx])
        # self.save1 = self.transformed_recon_face.clone()
        # self.save2 = self.transformed_recon_edge.clone()

        self.transformed_recon_face = apply_transform_batch(self.recon_face, self.model.surf_st)
        self.transformed_recon_edge = apply_transform_batch(self.recon_edge, self.model.edge_st)

        # assert torch.allclose(self.save1, self.transformed_recon_face)
        # assert torch.allclose(self.save2, self.transformed_recon_edge)

    def dist(self, pc1, pc2, point_reduction='mean'):
        return self.chamfer_dist(pc1.reshape(1, -1, 3), pc2.reshape(1, -1, 3), point_reduction=point_reduction)

    def option_loss(self):
        size_consistency_loss = 0
        wire_distance_loss = 0
        # wire_length_loss = 0
        for face_idx, face_edge_idx in enumerate(self.face_edge_adj):
            face_sc = get_size_and_center(self.transformed_recon_face[face_idx].reshape(1, -1, 3))[0]
            wire_sc = get_size_and_center(self.transformed_recon_edge[face_edge_idx].reshape(1, -1, 3))[0]
            size_consistency_loss += (face_sc[0] - wire_sc[0]) ** 2
            wire_distance_loss += self.dist(self.transformed_recon_edge[face_edge_idx], self.transformed_recon_face[face_idx])
            # wire_length_loss += get_wire_length(self.transformed_recon_edge[face_edge_idx])

        coin_edge_loss = self.chamfer_dist(self.transformed_recon_edge[self.coincident_edge_idx][:, 0, :, :],
                                           self.transformed_recon_edge[self.coincident_edge_idx][:, 1, :, :], batch_reduction='sum')

    def loss(self):
        adj_distance_loss = 0
        for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
            edge_to_face1 = self.dist(self.transformed_recon_edge[edge_idx], self.transformed_recon_face[face_idx1])
            edge_to_face2 = self.dist(self.transformed_recon_edge[edge_idx], self.transformed_recon_face[face_idx2])
            adj_distance_loss_c = (edge_to_face1 + edge_to_face2) / 2
            if adj_distance_loss_c > 0.01:
                adj_distance_loss += adj_distance_loss_c

        if self.penalized_vertex.shape[0] != 0:
            penalized_vertex_pair = self.transformed_recon_edge[:, [0, -1], :].reshape(-1, 3)[self.penalized_vertex]
            compat_vertexes_loss = torch.norm(penalized_vertex_pair[:, 0, :] - penalized_vertex_pair[:, 1, :], dim=-1).mean()
        else:
            compat_vertexes_loss = 0

        sum_loss = adj_distance_loss + compat_vertexes_loss
        return sum_loss

    def run(self):
        with tqdm(total=self.max_iter, desc='Geom Optimization', unit='iter') as pbar:
            for iter in range(self.max_iter):
                self.apply_all_transform()
                loss = self.loss()
                self.optimizer.zero_grad()
                if loss < 0.003:
                    print(f'Early stop at iter {iter}')
                    break
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_postfix(loss=loss.item())
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
    geom_opt = Geom_Optimization(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, max_iter=max_iter)
    geom_opt.run()
    return geom_opt.get_transformed_data()


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
