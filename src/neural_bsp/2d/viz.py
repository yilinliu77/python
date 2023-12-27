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

from shared.common_utils import check_dir, safe_check_dir


def normalize(v, v_axis=-1):
    norm = np.linalg.norm(v, axis=v_axis)
    new_v = np.copy(v)
    new_v[norm > 0] = new_v[norm > 0] / norm[norm > 0][:, None]
    return new_v


# Generate query points in [-0.5,0.5] in a resolution^3
def generate_query_points(v_resolution=100):
    query_points_x = np.linspace(-0.5, 0.5, v_resolution, dtype=np.float64)
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
    ], dtype=np.float64)
    return vertices


def sample_edges(v_vertices, v_resolution_meter=0.01):
    edges = np.stack((v_vertices, np.roll(v_vertices, -1, axis=0)), axis=1)
    edges = edges[:-1]
    dir = edges[:, 1] - edges[:, 0]
    points = []
    num_primitives = []
    for i_edge in range(v_vertices.shape[0] - 1):
        num_sampled = int(np.linalg.norm(dir[i_edge]) / v_resolution_meter)
        sample_base = np.linspace(0, 1, num_sampled, dtype=np.float64)
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


def generate_test_shape2(v_resolution_meter=0.01):
    ccube_vertices = cube_vertices()
    surface_points1, num_primitives1 = sample_edges(ccube_vertices, v_resolution_meter)
    sample_points1 = np.concatenate((ccube_vertices, surface_points1), axis=0)
    num_primitives1 = np.concatenate((np.ones(ccube_vertices.shape[0], dtype=np.int64), num_primitives1), axis=0)

    # x^2+0.1x+0.0025+y^2-0.01=0
    quadric_points1_x = np.linspace(-0.15, -0.05, int((-0.05 + 0.15) / v_resolution_meter), dtype=np.float64)
    quadric_points1_y = np.sqrt(10 * -(quadric_points1_x ** 2 + 0.2 * quadric_points1_x + 0.0001 - 0.01)) - 0.45
    sample_points2 = np.stack((quadric_points1_x, quadric_points1_y), axis=1)
    num_primitives2 = np.array((1, sample_points2.shape[0] - 2, 1), dtype=np.int64)
    # x^2-0.1x+0.0025+y^2-0.01=0
    quadric_points2_x = np.linspace(0.05, 0.15, int((0.15 - 0.05) / v_resolution_meter), dtype=np.float64)
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


def generate_test_shape3(v_resolution_meter=0.01):
    # Cube
    ccube_vertices = np.array([
        [0.05, 0.2], [-0.25, 0.2], [-0.25, -0.2], [0.25, -0.2], [0.25, 0.],
    ], dtype=np.float64)
    surface_points1, num_primitives1 = sample_edges(ccube_vertices, v_resolution_meter)
    sample_points1 = np.concatenate((ccube_vertices, surface_points1), axis=0)
    num_primitives1 = np.concatenate((np.ones(ccube_vertices.shape[0], dtype=np.int64), num_primitives1), axis=0)

    # x^2+0.1x+0.0025+y^2-0.01=0
    quadric_points1_x = np.linspace(0, np.pi / 2, int(np.pi / 2 / v_resolution_meter / 5), dtype=np.float64)
    quadric_points1_y = np.sin(quadric_points1_x)
    quadric_points1_x = np.cos(quadric_points1_x)
    sample_points2 = np.stack((quadric_points1_x, quadric_points1_y), axis=1)
    sample_points2 = sample_points2 * 0.2
    sample_points2[:,0]+=0.05
    num_primitives2 = np.array((sample_points2.shape[0],), dtype=np.int64)

    # x^2+0.1x+0.0025+y^2-0.01=0
    quadric_points2_x = np.linspace(0, 2 * np.pi, int(2 * np.pi / v_resolution_meter / 10), dtype=np.float64)
    quadric_points2_y = np.sin(quadric_points2_x)
    quadric_points2_x = np.cos(quadric_points2_x)
    quadric_points2_x = np.concatenate((quadric_points2_x, quadric_points2_x), axis=0)
    quadric_points2_y = np.concatenate((quadric_points2_y, -quadric_points2_y), axis=0)
    sample_points3 = np.stack((quadric_points2_x, -quadric_points2_y), axis=1)
    sample_points3 = sample_points3 * 0.09
    num_primitives3 = np.array((sample_points3.shape[0],), dtype=np.int64)

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
    kdtree.add(target_vertices.astype(np.float64))

    # The nearest surface points of all query points
    squared_dis, nearest_ids = kdtree.search(query_points, 1)
    nearest_surface_points = target_vertices[nearest_ids[:, 0],]
    udf = np.sqrt(squared_dis[:, 0])

    dir = normalize(nearest_surface_points - query_points, -1)

    return udf, nearest_ids[:, 0], dir


def visualize_raw_data(query_points, target_vertices, v_viz_flag):
    if not v_viz_flag:
        return

    colormap = np.asarray([
        (246,189,96),
        # (247,237,226),
        (210,180,140),
        (245,202,195),
        (132,165,157),
        (242,132,130)]
    )/255.

    colormap = np.asarray([
        (246,210,147),
        # (247,237,226),
        (210,192,168),
        (245,224,221),
        (150,166,163),
        (242,182,180)]
    )/255.

    N = 30
    segmented_cmaps = matplotlib.colors.ListedColormap([
        colormap[i%colormap.shape[0]] for i in range(N)
    ])

    plt.scatter(query_points[:, 0], query_points[:, 1], s=5, c=id_nearest_primitive,
                    # cmap="tab20"
                    cmap=segmented_cmaps
                    )
    # plt.title("Generalized space partition")
    # plt.colorbar()
    target_vertices = np.load("1.npy")
    plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    plt.axis('scaled')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
    plt.savefig("output/gsp_viz/4_gsp_gt.png", dpi=600, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()


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
    # plt.title("Boundary edges")
    # target_vertices = np.load("1.npy")
    # plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    x_values = [query_points[boundary_edges[:, 0], 0], query_points[boundary_edges[:, 1], 0]]
    y_values = [query_points[boundary_edges[:, 0], 1], query_points[boundary_edges[:, 1], 1]]
    plt.plot(x_values, y_values, '-', color=np.asarray((132,165,157))/255.)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.axis("scaled")
    # plt.show(block=True)
    plt.savefig("output/gsp_viz/5_gsp_boundary.png", dpi=600, bbox_inches='tight')
    plt.close()


def calculate_perpendicular_direction(g):
    g_h = np.concatenate((g, np.zeros_like(g[:, :, 0:1])), axis=2)
    up_h = np.zeros_like(g_h)
    up_h[:, :, 2] = 1
    gd = np.cross(up_h, g_h)[:, :, :2]
    gd = normalize(gd, -1)
    return gd


def calculate_gradients(v_query_points, v_min_dis, v_resolution):
    q = v_query_points.reshape(v_resolution, v_resolution, 2)
    d = v_min_dis.reshape(v_resolution, v_resolution, )

    # Calculate gradients
    gy, gx = np.gradient(d, )
    g = -np.stack((gx, gy), axis=-1)
    g = normalize(g, v_axis=-1)

    # # Calculate the perpendicular direction
    # g_h = np.concatenate((g, np.zeros_like(g[:, :, 0:1])), axis=2)
    # up_h = np.zeros_like(g_h)
    # up_h[:, :, 2] = 1
    # gd = np.cross(up_h, g_h)[:, :, :2]
    gd = calculate_perpendicular_direction(g)

    # Index the nearby gradient
    coords = np.stack(np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), indexing="xy"), axis=-1)
    interpolator = scipy.interpolate.RegularGridInterpolator((np.arange(v_resolution), np.arange(v_resolution)), g)

    coords_g = coords + gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < v_resolution - 1)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, v_resolution - 1)

    gn1 = interpolator(coords_g[:, :, ::-1])
    gn1[~valid_mask] = 0
    gn1 = normalize(gn1, v_axis=-1)

    coords_g = coords - gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < v_resolution - 1)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, v_resolution - 1)

    gn2 = interpolator(coords_g[:, :, ::-1])
    gn2[~valid_mask] = 0
    gn2 = normalize(gn2, v_axis=-1)

    return g, gn1, gn2, gd


def visualize_udf(query_points, udf, v_viz_flag):
    if not v_viz_flag:
        return

    color1 = np.asarray([245, 202, 195]) / 255
    color2 = np.asarray([132, 165, 157]) / 255
    udf1 = udf.max()
    udf2 = udf.min()
    level = 1-np.exp(-3 * (udf-udf2)/(udf1-udf2))
    plt.cm.register_cmap('my_cm',
                         matplotlib.colors.LinearSegmentedColormap.from_list('my_cm', [color1, color2]))

    plt.title("Unsigned distance field")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    plt.scatter(query_points[:, 0], query_points[:, 1], s=10, c=level, cmap="my_cm")
    plt.colorbar()
    plt.savefig("output/gsp_viz/0_udp.png", dpi=600, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()


def visualize_target_shape(target_vertices, v_viz_flag):
    if not v_viz_flag:
        return

    plt.title("Target shape")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    # plt.colorbar()
    plt.savefig("output/gsp_viz/0_target.png", dpi=600, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()


def visualize_3th_derivative(query_points, target_vertices, magic_gg, v_viz_flag):
    if not v_viz_flag:
        return

    color2 = np.asarray([245, 202, 195]) / 255
    color1 = np.asarray([132, 165, 157]) / 255
    udf1 = magic_gg.max()
    udf2 = magic_gg.min()
    level = (magic_gg - udf2) / (udf1 - udf2)
    plt.cm.register_cmap('my_cm1',
                         matplotlib.colors.LinearSegmentedColormap.from_list('my_cm1', [color1, color2]))

    plt.title("3rd derivative")
    plt.scatter(query_points[:, 0], query_points[:, 1], s=3, c=level, cmap="my_cm1")
    plt.colorbar()
    # plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=5, c="r")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig("output/gsp_viz/3_quantity.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)
    plt.close()

    return


def visualize_gradient_move(query_points, target_vertices, g, v_resolution, v_viz_flag):
    if not v_viz_flag:
        return
    g = g.reshape(-1, 2)
    color = np.stack(
        [(g[:, 0] + 1) / 2, (g[:, 1] + 1) / 2, np.zeros_like(g[:, 0])],
        axis=-1
    ).clip(0, 1)

    for i in range(3):
        g_ = np.sqrt(np.abs(g))
        g_[g < 0] = -g_[g < 0]
        g = g_
    # g = normalize(g, v_axis=-1)

    plt.title("2nd derivative (Direction change along the chosen direction)")
    # plt.scatter(target_vertices[:, 0], target_vertices[:, 1], color=(1, 0, 0), s=3)
    plt.quiver(query_points[:, 0], query_points[:, 1], g[:, 0], g[:, 1],
               angles='xy', scale_units='xy',
               scale=16,
               width=0.01,
               headwidth=4,
               headaxislength=4,
               headlength=4,
               color=color
               )
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig("output/gsp_viz/2_direction_move.png", dpi=600, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()

    return


def visualize_color_bar(v_viz_flag):
    if not v_viz_flag:
        return
    # Draw a circle with the color of each direction
    x_base = np.linspace(-np.pi, np.pi, 1000)
    x = np.cos(x_base)
    y = np.sin(x_base)

    points = []
    for r in np.arange(0.4, 0.5, 0.001):
        points.append((r * x, r * y))
    points = np.array(points)
    color = np.stack(
        [(x + 1) / 2, (y + 1) / 2, np.zeros_like(x)],
        axis=-1
    )
    color = np.repeat(color[np.newaxis, :, :], points.shape[0], axis=0).reshape(-1, 3)

    plt.axhline(0, color='red')
    plt.axvline(0, color='green')
    plt.scatter(points[:, 0], points[:, 1], s=5, color=color)
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.plot((1), (0), ls="", marker=">", ms=5, color="r",
             transform=plt.gca().get_yaxis_transform(), clip_on=False)
    plt.plot((0), (1), ls="", marker="^", ms=5, color="g",
             transform=plt.gca().get_xaxis_transform(), clip_on=False)

    # remove the border
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig("output/gsp_viz/color_bar.png", dpi=600, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()

    return


def visualize_gradient_direction(query_points, target_vertices, g, v_resolution, v_viz_flag, v_g_or_gn):
    if not v_viz_flag:
        return
    g = normalize(g.reshape(-1, 2), -1)

    color = np.stack(
        [(g[:, 0] + 1) / 2, (g[:, 1] + 1) / 2, np.zeros_like(g[:, 0])],
        axis=-1
    )

    plt.quiver(query_points[:, 0], query_points[:, 1], g[:, 0], g[:, 1],
               angles='xy', scale_units='xy',
               scale=16,
               width=0.01,
               headwidth=4,
               headaxislength=4,
               headlength=4,
               color=color
               )
    if v_g_or_gn:
        plt.title("1st derivative (gradient direction)")
    else:
        plt.title("Perpendicular direction")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.axis('scaled')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # plt.scatter(target_vertices[:, 0], target_vertices[:, 1], s=3, color=(1, 0, 0))
    if v_g_or_gn:
        plt.savefig("output/gsp_viz/1_direction.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig("output/gsp_viz/1_perpendicular.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)
    plt.close()

    return


if __name__ == '__main__':
    print("Start to construct dataset")
    np.random.seed(0)

    safe_check_dir("output/gsp_viz")

    first_round = True
    resolution = 256 if first_round else 16
    query_points = generate_query_points(v_resolution=resolution)
    # target_vertices, num_primitives = generate_test_shape2(v_resolution_meter=0.0001)
    target_vertices, num_primitives = generate_test_shape3(v_resolution_meter=0.00001)
    # target_vertices, num_primitives = generate_test_shape3(v_resolution_meter=0.03)
    id_primitives = np.arange(num_primitives.shape[0]).repeat(num_primitives)

    # Calculate distances
    udf, id_nearest_point, g = calculate_distances(query_points, target_vertices)
    id_nearest_primitive = id_primitives[id_nearest_point]

    visualize_target_shape(target_vertices, first_round)
    visualize_udf(query_points, udf, first_round)

    # 1st derivative
    gn = calculate_perpendicular_direction(g.reshape(resolution, resolution, 2))
    visualize_color_bar(True)
    visualize_gradient_direction(query_points, target_vertices, g, resolution, not first_round, True)
    visualize_gradient_direction(query_points, target_vertices, gn, resolution, not first_round, False)

    # Calculate the gradient of the vector field "g" along the direction "gn"
    # 2nd derivative
    g_y_2, g_x_2, _ = np.gradient(g.reshape(resolution, resolution, 2))
    gradient_move = g_x_2 * gn[:, :, 0:1] + g_y_2 * gn[:, :, 1:2]
    visualize_gradient_move(query_points, target_vertices, gradient_move, resolution, not first_round)

    # 3rd derivative
    magic_gg_y, magic_gg_x, _ = np.gradient(gradient_move)
    magic_gg = np.linalg.norm(magic_gg_x * gn[:, :, 0:1] + magic_gg_y * gn[:, :, 1:2], axis=-1)
    magic_gg = magic_gg / magic_gg.max()
    magic_gg = np.log(magic_gg + 1e-16).clip(-6, 0)
    visualize_3th_derivative(query_points, target_vertices, magic_gg, first_round)

    #
    visualize_raw_data(query_points, target_vertices, first_round)
    #
    all_edges, valid_flag = construct_graph(resolution, is_eight_neighbour=True)
    valid_edges = all_edges[valid_flag]
    #
    consistent_flag_ = id_nearest_primitive[valid_edges[:, 0]] == id_nearest_primitive[valid_edges[:, 1]]
    consistent_flag = np.ones_like(valid_flag)
    consistent_flag[valid_flag] = consistent_flag_
    boundary_edges = all_edges[~consistent_flag]

    visualize_boundary_edge(query_points, target_vertices, boundary_edges, first_round)

    input_features = np.concatenate((udf.reshape(-1, 1), g.reshape(-1, 2)), axis=1)

    np.save("output/gsp_edges_2d", {
        "resolution": resolution,
        "input_features": input_features,
        "all_edges": all_edges,
        "valid_flags": valid_flag,
        "consistent_flags": consistent_flag,
        "target_vertices": target_vertices,
        "query_points": query_points,
    })

    exit(0)
