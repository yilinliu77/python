import time

import torch

from shared.common_utils import normalize_tensor
from src.neural_recon.geometric_util import vectors_to_angles, intersection_of_ray_and_plane


# cur_dir: (M, 3) M directions
# start_point: (M, 3) M points
# num_per_edge_m: float value
def sample_edge(num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
    times = [0 for _ in range(10)]
    cur_time = time.time()
    length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
    num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
    num_edge_points_ = num_edge_points.roll(1)
    num_edge_points_[0] = 0
    times[1] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = torch.arange(num_edge_points.sum(), device=cur_dir.device) - num_edge_points_.cumsum(
        dim=0).repeat_interleave(num_edge_points)
    times[2] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
    times[3] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                          + start_point.repeat_interleave(num_edge_points, dim=0)
    times[4] += time.time() - cur_time
    cur_time = time.time()
    return num_edge_points, sampled_edge_points


# p1, p2, p3: (M, 3) M points
# p1: (M, 3) M points
# num_per_m: float value
def sample_triangles(num_per_m, p1, p2, p3, num_max_sample=500, v_sample_edge=True):
    d1 = p2 - p1
    d2 = p3 - p2
    area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

    num_per_m2 = num_per_m * num_per_m
    num_tri_samples = torch.clamp((area * num_per_m2).to(torch.long), 1, num_max_sample * 4)

    # samples = torch.rand(num_tri_samples.sum(), 2, device=p1.device)
    g = torch.Generator(device=p1.device)
    g.manual_seed(0)
    samples = torch.rand(num_tri_samples.sum(), 2, device=p1.device, generator=g)
    u = (p2 - p1).repeat_interleave(num_tri_samples, dim=0)
    v = (p3 - p1).repeat_interleave(num_tri_samples, dim=0)
    sampled_polygon_points = samples[:, 0:1] * u + samples[:, 1:2] * v
    inside_mask = samples.sum(dim=-1) > 1
    sampled_polygon_points[inside_mask] = ((1 - samples[:, 0:1]) * u + (1 - samples[:, 1:2]) * v)[inside_mask]
    sampled_polygon_points = sampled_polygon_points + p1.repeat_interleave(num_tri_samples, dim=0)

    if v_sample_edge:
        num_edge_points, edge_points = sample_edge(num_per_m,
                                                   torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                   # torch.stack((d1,), dim=1).reshape(-1, 3),
                                                   torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                   # torch.stack((p1,), dim=1).reshape(-1, 3),
                                                   num_max_sample=num_max_sample)
        num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)
        num_total_points = num_edge_points + num_tri_samples
        num_total_points_cumsum = num_total_points.cumsum(0).roll(1)
        num_total_points_cumsum[0] = 0
        sampled_total_points = torch.zeros((num_total_points.sum(), 3), device=p1.device, dtype=torch.float32)
        num_edge_points_ = num_edge_points.cumsum(0).roll(1)
        num_edge_points_[0] = 0
        num_tri_points_ = num_tri_samples.cumsum(0).roll(1)
        num_tri_points_[0] = 0
        edge_index = torch.arange(num_edge_points.sum(), device=p1.device) \
                     - (num_edge_points_ - num_total_points_cumsum).repeat_interleave(num_edge_points)
        tri_index = torch.arange(num_tri_samples.sum(), device=p1.device) \
                    - (num_tri_points_ - num_total_points_cumsum - num_edge_points).repeat_interleave(
            num_tri_samples)
        sampled_total_points[edge_index] = edge_points
        sampled_total_points[tri_index] = sampled_polygon_points
        return num_total_points, sampled_total_points
    else:
        return num_tri_samples, sampled_polygon_points

# v_edge_points: (M, 2, 3) M end points pairs
# v_num_horizontal: (M, 1) The number of the horizontal points for each edge
def sample_points_2d(v_edge_points, v_num_horizontal,
                     v_img_width=800, v_vertical_length=10,
                     v_max_points=500):
    device = v_edge_points.device
    cur_dir = v_edge_points[:, 1] - v_edge_points[:, 0]
    cur_length = torch.linalg.norm(cur_dir, dim=-1) + 1e-6

    cur_dir_h = torch.cat((cur_dir, torch.zeros_like(cur_dir[:, 0:1])), dim=1)
    z_axis = torch.zeros_like(cur_dir_h)
    z_axis[:, 2] = 1
    edge_up = normalize_tensor(torch.cross(cur_dir_h, z_axis, dim=1)[:, :2]) * v_vertical_length / v_img_width
    # The vertical length is 10 -> 10/6000 = 0.00167

    # Compute interpolated point
    num_horizontal = v_num_horizontal
    num_half_vertical = 10
    num_coordinates_per_edge = num_horizontal * num_half_vertical * 2

    begin_idxes = num_horizontal.cumsum(dim=0)
    total_num_x_coords = begin_idxes[-1]
    begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
    begin_idxes[0] = 0  # (M,)
    dx = torch.arange(num_horizontal.sum(), device=device) - \
         begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
    dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal)
    dy = torch.arange(num_half_vertical, device=device) / (num_half_vertical - 1)
    dy = torch.cat((torch.flip(-dy, dims=[0]), dy))

    # Meshgrid
    total_num_coords = total_num_x_coords * dy.shape[0]
    coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * dy.shape[0])  # (total_num_coords,)
    coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
    coords = torch.stack((coords_x, coords_y), dim=1)

    interpolated_coordinates = \
        cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
        edge_up.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
        v_edge_points[:, 0].repeat_interleave(num_coordinates_per_edge, dim=0)

    return num_coordinates_per_edge, interpolated_coordinates

# v_original_distances: (M, 1) M original samples
def sample_new_distance(v_original_distances,
                        num_sample=100, scale_factor=1.0,
                        v_max=10., v_min=0., v_random_g=None):
    num_vertices = v_original_distances.shape[0]
    device = v_original_distances.device
    # (B, S)
    new_distance = -torch.ones((num_vertices, num_sample - 1), device=device, dtype=v_original_distances.dtype)
    sample_distance_mask = torch.zeros_like(new_distance).to(torch.bool)
    # (B, S)
    repeated_vertices_distances = v_original_distances[:, None].tile((1, num_sample - 1))

    while not torch.all(sample_distance_mask):
        t_ = new_distance[~sample_distance_mask]
        a = repeated_vertices_distances[~sample_distance_mask] + \
            scale_factor * torch.normal(
            torch.zeros(t_.shape[0], dtype=t_.dtype, device=device),
            torch.ones(t_.shape[0], dtype=t_.dtype, device=device), generator=v_random_g)
        new_distance[~sample_distance_mask] = a
        sample_distance_mask = torch.logical_and(new_distance > v_min, new_distance < v_max)
    # (B, (S + 1))
    new_distance = torch.cat((v_original_distances[:, None], new_distance), dim=1)
    return new_distance


def sample_depth_and_angle(depth, angle, num_sample=100, v_random_g=None):
    # sample depth
    sample_depths = sample_new_distance(depth, num_sample, v_random_g=v_random_g)
    sample_angles = sample_new_distance(angle.reshape(-1), num_sample, scale_factor=torch.pi / 3, v_max=100,
                                        v_min=-100, v_random_g=v_random_g)
    sample_angles = sample_angles.reshape(depth.shape[0], 2, num_sample) % (2 * torch.pi)
    return sample_depths, sample_angles.permute(0, 2, 1)


def sample_new_planes(v_original_parameters, v_centroid_rays_c, v_dual_graph=None, v_random_g=None):
    plane_angles = vectors_to_angles(v_original_parameters[:, :3])
    initial_centroids = intersection_of_ray_and_plane(v_original_parameters, v_centroid_rays_c)[1]
    init_depth = torch.linalg.norm(initial_centroids, dim=-1)

    sample_depth, sample_angle = sample_depth_and_angle(init_depth, plane_angles, 100, v_random_g=v_random_g)

    if v_dual_graph is None:
        return sample_depth.contiguous(), sample_angle.contiguous()

    id_neighbour_patches = [list(v_dual_graph[id_node].keys()) for id_node in v_dual_graph.nodes]
    for patch_id in range(len(id_neighbour_patches)):
        # propagation from neighbour, do not sample depth!
        id_neighbour = id_neighbour_patches[patch_id]
        sample_depth[patch_id, 1:1 + len(id_neighbour)] = init_depth[torch.tensor(id_neighbour)]
        sample_angle[patch_id, 1:1 + len(id_neighbour)] = plane_angles[torch.tensor(id_neighbour)].clone()
    return sample_depth.contiguous(), sample_angle.contiguous()
