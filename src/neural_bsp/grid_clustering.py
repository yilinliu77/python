import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
import ray
import scipy
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import faiss
import torch
from tqdm import tqdm

from scipy.spatial import Delaunay, distance_matrix


def normalize(v, v_axis):
    norm = np.linalg.norm(v, axis=v_axis)
    new_v = np.copy(v)
    new_v[norm > 0] = new_v[norm > 0] / norm[norm > 0][:, None]
    return new_v


# Generate query points in [-0.5,0.5] in a resolution^3
def generate_query_points(v_resolution=100):
    query_points_x = np.linspace(-0.5, 0.5, v_resolution, dtype=np.float32)
    query_points = np.stack(np.meshgrid(query_points_x, query_points_x), axis=2)
    query_points = query_points.reshape(-1, 2)
    return query_points


# A concave shape
#
#  __    __
# |  |__|  |
# |        |
# |________|
#
def cube_vertices():
    vertices = np.array([
        [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25], [0.25, 0.25],
        [0.1, 0.25], [0.1, 0.1], [-0.1, 0.1], [-0.1, 0.25]
    ], dtype=np.float32)
    return vertices


def sample_edges(v_vertices):
    edges = np.stack((v_vertices, np.roll(v_vertices, -1, axis=0)), axis=1)
    dir = edges[:, 1] - edges[:, 0]
    points = []
    num_primitives = []
    for i_edge in range(v_vertices.shape[0]):
        num_sampled = int(np.linalg.norm(dir[i_edge]) / 0.01)
        sample_base = np.linspace(0, 1, num_sampled)
        sample_base = np.delete(sample_base, 0)
        sample_base = np.delete(sample_base, -1)
        points.append(edges[i_edge, 0] + dir[i_edge] * sample_base[:, None])
        num_primitives.append(sample_base.shape[0])
    points = np.concatenate(points, axis=0)
    return points, np.array(num_primitives, np.int64)


def generate_test_shape1():
    ccube_vertices = cube_vertices()
    surface_points = sample_edges(ccube_vertices)
    return surface_points


def generate_test_shape2():
    ccube_vertices = cube_vertices()
    surface_points1, num_primitives1 = sample_edges(ccube_vertices)
    sample_points1 = np.concatenate((ccube_vertices, surface_points1), axis=0)
    num_primitives1 = np.concatenate((np.ones(ccube_vertices.shape[0], dtype=np.int64), num_primitives1), axis=0)

    # x^2+0.1x+0.0025+y^2-0.01=0
    quadric_points1_x = np.linspace(-0.15, -0.05, 50)
    quadric_points1_y = np.sqrt(10 * -(quadric_points1_x ** 2 + 0.2 * quadric_points1_x + 0.0001 - 0.01)) - 0.45
    sample_points2 = np.stack((quadric_points1_x, quadric_points1_y), axis=1)
    num_primitives2 = np.array((1, sample_points2.shape[0] - 2, 1), dtype=np.int64)
    # x^2-0.1x+0.0025+y^2-0.01=0
    quadric_points2_x = np.linspace(0.05, 0.15, 50)
    quadric_points2_y = np.sqrt(10 * -(quadric_points2_x ** 2 - 0.2 * quadric_points2_x + 0.0001 - 0.01)) - 0.45
    sample_points3 = np.stack((quadric_points2_x, quadric_points2_y), axis=1)
    num_primitives3 = np.array((1, sample_points3.shape[0] - 2, 1), dtype=np.int64)

    points = np.concatenate((
        sample_points1,
        sample_points2,
        sample_points3,
    ), axis=0)
    num_primitives = np.concatenate((
        num_primitives1,
        num_primitives2,
        num_primitives3,
    ), axis=0)

    return points, num_primitives


def calculate_distances(query_points, target_vertices):
    kdtree = faiss.IndexFlatL2(2)
    res = faiss.StandardGpuResources()
    kdtree = faiss.index_cpu_to_gpu(res, 0, kdtree)
    kdtree.add(target_vertices.astype(np.float32))

    # The nearest surface points of all query points
    squared_dis, nearest_ids = kdtree.search(query_points, 1)
    nearest_surface_points = target_vertices[nearest_ids[:, 0],]
    udf = np.sqrt(squared_dis[:, 0])
    return udf, nearest_ids[:, 0]


def visualize_raw_data(query_points, target_vertices, v_viz_flag):
    if not v_viz_flag:
        return
    # Visualize
    plt.scatter(query_points[:, 0], query_points[:, 1], s=5, c=id_nearest_primitive,
                cmap="tab20"
                # cmap="flag"
                )
    plt.colorbar()
    plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))

    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.axis('scaled')
    plt.show(block=True)


# Construct the edge graph
def construct_graph(v_resolution, is_eight_neighbour):
    all_edges = []
    valid_flag = []
    for y in range(v_resolution):
        for x in range(v_resolution):
            all_edges.append(((x, y), (x - 1, y - 1)))
            all_edges.append(((x, y), (x, y - 1)))
            all_edges.append(((x, y), (x + 1, y - 1)))
            all_edges.append(((x, y), (x + 1, y)))
            all_edges.append(((x, y), (x + 1, y + 1)))
            all_edges.append(((x, y), (x, y + 1)))
            all_edges.append(((x, y), (x - 1, y + 1)))
            all_edges.append(((x, y), (x - 1, y)))
            if y > 0 and x > 0 and is_eight_neighbour:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if y > 0:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if y > 0 and x < v_resolution - 1 and is_eight_neighbour:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if x < v_resolution - 1:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if x < v_resolution - 1 and y < v_resolution - 1 and is_eight_neighbour:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if y < v_resolution - 1:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if y < v_resolution - 1 and x > 0 and is_eight_neighbour:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
            if x > 0:
                valid_flag.append(True)
            else:
                valid_flag.append(False)
    all_edges = np.array(all_edges)
    all_edges = all_edges[:, :, 0] + all_edges[:, :, 1] * v_resolution
    valid_flag = np.array(valid_flag)
    return all_edges, valid_flag


def visualize_boundary_edge(query_points, target_vertices, boundary_edges, v_viz_flag):
    if not v_viz_flag:
        return
    plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    x_values = [query_points[boundary_edges[:, 0], 0], query_points[boundary_edges[:, 1], 0]]
    y_values = [query_points[boundary_edges[:, 0], 1], query_points[boundary_edges[:, 1], 1]]
    plt.plot(x_values, y_values, 'g-')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.axis("scaled")
    plt.show(block=True)


def calculate_gradients(v_query_points, v_min_dis, v_resolution):
    q = v_query_points.reshape(v_resolution, v_resolution, 2)
    d = v_min_dis.reshape(v_resolution, v_resolution, )

    # Calculate gradients
    gy, gx = np.gradient(d, )
    g = -np.stack((gx, gy), axis=-1)
    g = normalize(g, v_axis=-1)

    # Calculate the perpendicular direction
    g_h = np.concatenate((g, np.zeros_like(g[:, :, 0:1])), axis=2)
    up_h = np.zeros_like(g_h)
    up_h[:, :, 2] = 1
    gd = np.cross(up_h, g_h)[:, :, :2]
    gd = normalize(gd, -1) * 0.5

    # Index the nearby gradient
    coords = np.stack(np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), indexing="xy"), axis=-1)
    interpolator = scipy.interpolate.RegularGridInterpolator((np.arange(v_resolution), np.arange(v_resolution)), g)

    coords_g = coords + gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < v_resolution-1)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, v_resolution-1)

    gn1 = interpolator(coords_g[:, :, ::-1])
    gn1[~valid_mask] = 0
    gn1 = normalize(gn1, v_axis=-1)

    coords_g = coords - gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < v_resolution-1)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, v_resolution-1)

    gn2 = interpolator(coords_g[:, :, ::-1])
    gn2[~valid_mask] = 0
    gn2 = normalize(gn2, v_axis=-1)

    return g, gn1, gn2, gd


if __name__ == '__main__':
    print("Start to construct dataset")
    np.random.seed(0)

    resolution = 128
    query_points = generate_query_points(v_resolution=resolution)
    target_vertices, num_primitives = generate_test_shape2()
    id_primitives = np.arange(num_primitives.shape[0]).repeat(num_primitives)

    # Calculate distances
    udf, id_nearest_point = calculate_distances(query_points, target_vertices)
    id_nearest_primitive = id_primitives[id_nearest_point]

    g, gn1, gn2, gd = calculate_gradients(query_points, udf, resolution)

    #
    visualize_raw_data(query_points, target_vertices, False)
    #
    all_edges, valid_flag = construct_graph(resolution, is_eight_neighbour=True)
    valid_edges = all_edges[valid_flag]
    #
    consistent_flag_ = id_nearest_primitive[valid_edges[:, 0]] == id_nearest_primitive[valid_edges[:, 1]]
    consistent_flag = np.ones_like(valid_flag)
    consistent_flag[valid_flag] = consistent_flag_
    boundary_edges = all_edges[~consistent_flag]

    visualize_boundary_edge(query_points, target_vertices, boundary_edges, True)

    input_features = np.concatenate((udf.reshape(-1, 1), g.reshape(-1, 2)), axis=1)

    np.save("output/edges", {
        "resolution":resolution,
        "input_features":input_features,
        "all_edges":all_edges,
        "valid_flags":valid_flag,
        "consistent_flags":consistent_flag,
        "target_vertices":target_vertices,
        "query_points":query_points,
    })

    exit(0)
