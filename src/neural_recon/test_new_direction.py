import math

import open3d as o3d
import numpy as np
import torch

from shared.common_utils import *


def sample_new_directions_batch(v_input, num_sample, v_variance):
    batch_size = v_input.shape[0]
    device = v_input.device
    v_input = v_input / (v_input.norm(dim=1, keepdim=True)+1e-6)

    # Generate N random unit vectors for each input vector
    random_vectors = torch.randn(batch_size, num_sample, 3, device=device)
    random_vectors = random_vectors / (torch.norm(random_vectors, dim=2, keepdim=True) + 1e-6)

    # Project random vectors onto the plane orthogonal to the input vectors
    projections = torch.matmul(random_vectors, v_input.unsqueeze(2))
    orthogonal_vectors = random_vectors - projections * v_input.unsqueeze(1)

    # Generate Gaussian distributed angles
    angles = torch.randn(batch_size, num_sample, device=device) * math.sqrt(v_variance)

    # Calculate the new directions
    new_directions = torch.cos(angles).unsqueeze(2) * v_input.unsqueeze(1) + torch.sin(angles).unsqueeze(
        2) * orthogonal_vectors

    return new_directions


def visualize_vectors_with_arrows(origin, vectors):
    arrow_length = 1.0
    arrow_radius = 0.05
    cylinder_radius = 0.025

    arrows = o3d.geometry.TriangleMesh()
    for v in vectors:
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.03, cone_radius=0.05,
                                                       cylinder_height=0.7, cone_height=0.1,
                                                       resolution=5, cylinder_split=3)
        arrow.rotate(caculate_align_mat(v), center=(0, 0, 0))
        arrow.translate(origin)
        arrows += arrow
        colors = np.zeros_like(np.asarray(arrows.vertices))
        colors[:, 1] = 1
        arrows.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([arrows,o3d.geometry.TriangleMesh.create_coordinate_frame(2)])
    # o3d.visualization.draw_geometries([arrows])

# Example usage:
a = torch.tensor([1.0, 1.0, -1.0])[None,:]
N = 1000
m = 0.0
v = 0.3

new_directions = sample_new_directions_batch(a, N, v)[0]
print(new_directions)

origin = torch.tensor([0.0, 0.0, 0.0])

visualize_vectors_with_arrows(origin, new_directions)