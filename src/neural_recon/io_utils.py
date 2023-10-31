import random

import numpy as np
import torch

from shared.common_utils import ray_line_intersection2


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def save_plane(v_abcds, rays_c, patch_vertexes_id,
               file_path="output/init.ply"):
    vertices = []
    polygons = []
    acc_num_vertices = 0
    for id_patch in range(len(patch_vertexes_id)):
        intersection_points = ray_line_intersection2(v_abcds[id_patch],
                                                     torch.zeros_like(rays_c[patch_vertexes_id[id_patch]]),
                                                     rays_c[patch_vertexes_id[id_patch]])
        vertices.append(intersection_points)
        polygons.append(np.arange(intersection_points.shape[0]) + acc_num_vertices)
        acc_num_vertices += intersection_points.shape[0]
        pass
    vertices = torch.cat(vertices, dim=0).cpu().numpy()
    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\nproperty float x\nproperty float y\nproperty float z\n".format(
            acc_num_vertices))
        f.write("element face {}\nproperty list uchar int vertex_index\n".format(len(polygons)))
        f.write("end_header\n")
        for ver in vertices:
            f.write("{} {} {}\n".format(ver[0], ver[1], ver[2]))
        for polygon in polygons:
            f.write("{}".format(len(polygon)))
            for item in polygon:
                f.write(" {}".format(item))
            f.write("\n")
        pass
    print("Save done")
    return
