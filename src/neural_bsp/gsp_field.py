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


def sample_edges(v_vertices, v_resolution):
    edges = np.stack((v_vertices, np.roll(v_vertices, -1, axis=0)), axis=1)
    dir = edges[:, 1] - edges[:, 0]
    points = []
    id_primitives = []
    for i_edge in range(v_vertices.shape[0]):
        num_sampled = int(np.linalg.norm(dir[i_edge]) / (1/v_resolution))
        points.append(edges[i_edge, 0] + dir[i_edge] * np.linspace(0, 1, num_sampled)[:, None])
        id_primitives.append(num_sampled)
    points = np.concatenate(points, axis=0)
    return points, np.array(id_primitives, np.int64)


def generate_test_shape1():
    ccube_vertices = cube_vertices()
    surface_points = sample_edges(ccube_vertices)
    return surface_points


def generate_test_shape2(v_resolution = 50):
    ccube_vertices = cube_vertices()
    surface_points1, id_primitives = sample_edges(ccube_vertices, v_resolution)

    # x^2+0.1x+0.0025+y^2-0.01=0
    sphere_points1_x = np.linspace(-0.15, -0.05, int(v_resolution / 10) )
    sphere_points1_y = np.sqrt(10 * -(sphere_points1_x ** 2 + 0.2 * sphere_points1_x + 0.0001 - 0.01)) - 0.45
    # x^2-0.1x+0.0025+y^2-0.01=0
    sphere_points2_x = np.linspace(0.05, 0.15, int(v_resolution / 10) )
    sphere_points2_y = np.sqrt(10 * -(sphere_points2_x ** 2 - 0.2 * sphere_points2_x + 0.0001 - 0.01)) - 0.45

    points = np.concatenate((
        surface_points1,
        np.stack((sphere_points1_x, sphere_points1_y), axis=-1),
        np.stack((sphere_points2_x, sphere_points2_y), axis=-1),
    ))

    return points, np.concatenate((id_primitives, np.array((50, 50), np.int64)))


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
    coords = np.stack(np.meshgrid(np.arange(100), np.arange(100), indexing="xy"), axis=-1)
    interpolator = scipy.interpolate.RegularGridInterpolator((np.arange(100), np.arange(100)), g)

    coords_g = coords + gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < 99)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, 99)

    gn1 = interpolator(coords_g[:, :, ::-1])
    gn1[~valid_mask] = 0
    gn1 = normalize(gn1, v_axis=-1)

    coords_g = coords - gd
    valid_mask = np.logical_and(coords_g > 0, coords_g < 99)
    valid_mask = np.all(valid_mask, axis=2)
    valid_mask[0] = valid_mask[-1] = valid_mask[:, 0] = valid_mask[:, -1] = False
    coords_g = np.clip(coords_g, 0, 99)

    gn2 = interpolator(coords_g[:, :, ::-1])
    gn2[~valid_mask] = 0
    gn2 = normalize(gn2, v_axis=-1)

    return g, gn1, gn2, gd


def fit_curve(input_x, input_y):
    D1 = np.vstack([input_x ** 2, input_x * input_y, input_y ** 2]).T
    D2 = np.vstack([input_x, input_y, np.ones(len(input_x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeff = np.concatenate((ak, T @ ak)).ravel()
    return coeff


def fit_curve2(input_x, input_y):
    A = np.stack([input_x * input_x, input_x * input_y, input_y * input_y, input_x, input_y], axis=1)
    b = -np.ones(input_x.shape[0])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.insert(coeffs, 5, 1), residuals / input_x.shape[0]


def get_curve_points(coeff, x):
    A, B, C, D, E, F = coeff
    output_y1 = (-B * x - E - np.sqrt(
        np.clip((B * x + E) * (B * x + E) - 4 * C * (A * x * x + D * x + F), 0, np.inf))) / (2 * C)
    output_y2 = (-B * x - E + np.sqrt(
        np.clip((B * x + E) * (B * x + E) - 4 * C * (A * x * x + D * x + F), 0, np.inf))) / (2 * C)
    # y_ = np.stack(
    #     (coeff[2] * np.ones_like(input_x),
    #      coeff[1] * input_x + coeff[4],
    #      coeff[0] * input_x ** 2 + coeff[3] * input_x + 1), axis=1
    # )
    # output_y = np.asarray([np.roots(item) for item in y_], dtype=np.float32)
    return np.stack((output_y1, output_y2), axis=-1)


def fit_segment(input_x, input_y):
    k, b = np.polyfit(input_x, input_y, 1)
    # y=kx+b
    return np.array((k, -1, b))


def fit_segment_torch(batched_input):
    batched_input = torch.from_numpy(batched_input).cuda()
    A = torch.stack((batched_input[:, :, 0], torch.ones_like(batched_input[:, :, 0])), dim=2)
    kb = torch.linalg.lstsq(A, batched_input[:, :, 1]).solution
    abc = torch.stack((kb[:, 0], -torch.ones_like(kb[:, 0]), kb[:, 1]), dim=1)
    # y=kx+b
    return abc


def get_segment_points(coeff, input_x):
    output_y = -(input_x * coeff[0] + coeff[2]) / coeff[1]
    return output_y


def sample_edge(v_points, num_per_edge_m=100):
    v_segment = v_points[:, 1] - v_points[:, 0]
    length = np.linalg.norm(v_segment + 1e-6, axis=1)
    num_edge_points = np.clip((length * num_per_edge_m).astype(np.int64), 1, 2000)
    num_edge_points_ = np.roll(num_edge_points, 1)
    num_edge_points_[0] = 0
    sampled_edge_points = np.arange(num_edge_points.sum()) - num_edge_points_.cumsum(axis=0).repeat(num_edge_points)
    sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat(num_edge_points))
    sampled_edge_points = v_segment.repeat(num_edge_points, axis=0) * sampled_edge_points[:, None] \
                          + v_points[:, 0].repeat(num_edge_points, axis=0)
    return num_edge_points, sampled_edge_points


def find_level_set(v_query_points, v_dis, v_g, v_specific_distance=None, v_resolution=100):
    # v_query_points + v_g

    if v_specific_distance is not None:
        return v_query_points[np.where(np.abs(v_dis - v_specific_distance) < 0.005)]
    else:
        raise
    pass


@ray.remote
def compute_fitness(v_data, v_graph):
    results = []
    for id_start, id_end in v_data:
        if not nx.has_path(v_graph, id_start, id_end):
            results.append(1)
            continue
        paths = nx.shortest_path(v_graph, source=id_start, target=id_end)  # A list of the node is along the path
        if len(paths) < 5:
            results.append(0)
            continue

        path_length = len(paths)
        positions = np.stack([v_graph.nodes[idx]["pos"] for idx in paths], axis=0)
        fitted_curve, residual = fit_curve2(positions[:, 0], positions[:, 1])

        curve_y = get_curve_points(fitted_curve, positions[:, 0])
        true_y = positions[:, 1]
        calculated_error = np.abs(true_y[:, None] - curve_y).min(axis=1).mean()

        if False:
            print(np.sqrt(residual[0]) if residual.shape[0] > 0 else 0)
            print(calculated_error)
            print(id_start)
            # plt.scatter(grid_coords[:, 0], grid_coords[:, 1], s=1, color=(0,0,1))
            plt.axis('scaled')
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            plt.scatter(positions[:, 0], true_y, color=(0, 1, 0), s=2)
            plt.scatter(positions[:, 0], curve_y[:, 0], color=(1, 0, 0), s=1)
            plt.scatter(positions[:, 0], curve_y[:, 1], color=(1, 0, 0), s=1)
            plt.show(block=True)
        if residual.shape[0] == 0:
            results.append(0)
        else:
            error = np.abs(true_y[:, None] - curve_y).min(axis=1).mean()
            # error = min(np.sqrt(residual[0]) / 5, 1)
            results.append(error)
    return np.stack(results, axis=0)


def generate_training_data():
    query_points = generate_query_points()
    target_vertices, id_primitives = generate_test_shape2()

    kdtree = faiss.IndexFlatL2(2)
    res = faiss.StandardGpuResources()
    kdtree = faiss.index_cpu_to_gpu(res, 0, kdtree)
    kdtree.add(target_vertices.astype(np.float32))

    # The nearest surface points of all query points
    squared_dis, nearest_ids = kdtree.search(query_points, 1)
    nearest_surface_points = target_vertices[nearest_ids[:, 0],]

    udf = np.sqrt(squared_dis[:, 0])

    # Visualize the target shape
    if True:
        _, axes = plt.subplots(1, 2)
        axes[0].scatter(target_vertices[:, 0], target_vertices[:, 1], s=1, c=(1, 0, 0))
        axes[0].set_title("Target shape")
        axes[0].axis('scaled')
        axes[0].set_xlim(-0.5, 0.5)
        axes[0].set_ylim(-0.5, 0.5)
        axes[0].tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
        axes[1].scatter(query_points[:, 0], query_points[:, 1], c=udf)
        axes[1].set_title("UDF")
        axes[1].axis('scaled')
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_ylim(-0.5, 0.5)
        axes[1].tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)

        plt.show(block=True)

    g, _, _, _ = calculate_gradients(query_points, udf, v_resolution=100)
    g = normalize(g, v_axis=-1)
    coords_x = query_points.reshape(100, 100, 2)[0, :, 0]

    d_inter = scipy.interpolate.RegularGridInterpolator((coords_x, coords_x),
                                                        udf.reshape(100, 100, 1).transpose([1, 0, 2]),
                                                        bounds_error=False, fill_value=0)
    g_inter = scipy.interpolate.RegularGridInterpolator((coords_x, coords_x),
                                                        g.transpose([1, 0, 2]),
                                                        bounds_error=False, fill_value=0)

    # level_set_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    level_set_values = [0.03, ]
    level_set_colors = np.random.random((len(level_set_values), 3))
    graphs: list[nx.Graph] = []
    all_points = []
    for i_level, level_set_value in enumerate(level_set_values):
        # 1. Get all the level set points
        level_set_points = query_points + g.reshape(-1, 2) * (udf - level_set_value)[:, None]
        for i in range(5):
            d = d_inter(level_set_points)
            gradient_direction = normalize(g_inter(level_set_points), v_axis=-1)
            level_set_points = level_set_points + gradient_direction * (d - level_set_value)

        # Filter out
        level_set_points = level_set_points[np.abs(d_inter(level_set_points)[:, 0] - level_set_value) < 1e-6]

        # 2. Subsample the level set points
        grid_size = 0.01
        x_min, y_min = -0.5, -0.5
        x_max, y_max = +0.5, +0.5
        grid_width = int(np.ceil((x_max - x_min) / grid_size))
        grid_height = int(np.ceil((y_max - y_min) / grid_size))

        i_indices = ((level_set_points[:, 1] - y_min) / grid_size).astype(int)
        j_indices = ((level_set_points[:, 0] - x_min) / grid_size).astype(int)

        grid_sum = np.zeros((grid_height, grid_width, 2))
        grid_num = np.zeros((grid_height, grid_width), dtype=np.int64)
        np.add.at(grid_num, (i_indices, j_indices), 1)
        np.add.at(grid_sum, (i_indices, j_indices), level_set_points)
        grid_mean = grid_sum / grid_num[:, :, None]

        # 3. Subsample the level set points
        grid_coords = grid_mean[~np.isnan(grid_mean).all(axis=2)]

        gradient_direction = normalize(g_inter(grid_coords), v_axis=-1)
        distances = distance_matrix(grid_coords, grid_coords)
        for id_point in tqdm(range(distances.shape[0])):
            distances[id_point][distances[id_point] > 0.02] = np.inf
            dot_g = (gradient_direction[id_point] * gradient_direction).sum(axis=-1)
            distances[id_point][dot_g < 0] = np.inf

        # 4. Build the graph
        adj = distances.copy()
        adj[adj != np.inf] = 1
        adj[adj == np.inf] = 0
        np.fill_diagonal(adj, 0)
        graph = nx.from_numpy_array(adj)
        for id_node in graph.nodes:
            graph.nodes[id_node]["pos"] = grid_coords[id_node]

        components = list(nx.connected_components(graph))
        component_colors = np.random.random((len(components), 3))
        for i_component, component in enumerate(components):
            # Create a subgraph for the current component
            subgraph = graph.subgraph(component)
            for edge in subgraph.edges():
                # plt.plot(grid_coords[edge, 0], grid_coords[edge, 1], '-', c=component_colors[i_component])
                # plt.plot(grid_coords[edge, 0], grid_coords[edge, 1], '-', c=level_set_colors[i_level])
                pass

        all_points.append(grid_coords)
        graphs.append(graph)
        # for edge in graph.edges():
        #     plt.plot(grid_coords[edge, 0], grid_coords[edge, 1], 'r-')
        # plt.scatter(grid_coords[:, 0], grid_coords[:, 1], s=1)
        # plt.title("d={}".format(level_set_value))
        # plt.show(block=True)
    # plt.axis('scaled')
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(-0.5, 0.5)
    # plt.tick_params(left=False, right=False, labelleft=False,
    #                     labelbottom=False, bottom=False)
    # plt.show(block=True)

    # nearest_points_for_target_points_ = kdtree.search(target_vertices, 10)
    # nearest_points_for_target_points = target_vertices[nearest_points_for_target_points_[1]]
    # abcs = fit_segment_torch(nearest_points_for_target_points)

    num_nodes = graphs[0].number_of_nodes()
    all_combinations = torch.stack(
        torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes)), dim=2).view(-1,2).numpy()

    # Only for test
    # compute_fitness(all_combinations[147075:147629, ], graphs[0])

    if not os.path.exists("output/lsp.npy"):
        ray.init(
            # local_mode=True,
            num_cpus=24
        )
        tasks = []
        for i in range(24):
            id_start = (all_combinations.shape[0] // 24 + 1) * i
            id_end = (all_combinations.shape[0] // 24 + 1) * (i + 1)
            id_end = min(all_combinations.shape[0], id_end)
            tasks.append(compute_fitness.remote(
                all_combinations[id_start:id_end, ], graphs[0]))
            pass

        results = np.concatenate(ray.get(tasks), axis=0)
        ray.shutdown()
        np.save("output/lsp", results)
    else:
        results = np.load("output/lsp.npy")

    # Visualize
    if False:
        for _ in tqdm(range(100)):
            id_visualize = np.random.randint(0, all_points[0].shape[0])
            fig, axes = plt.subplots(1, 2)
            axes[0].set_xlim(-0.5, 0.5)
            axes[0].set_ylim(-0.5, 0.5)

            axes[0].scatter(level_set_points[:, 0], level_set_points[:, 1], s=1)
            axes[0].scatter(all_points[0][id_visualize, 0], all_points[0][id_visualize, 1], s=2, color=(1, 0, 0))
            axes[0].tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False)
            axes[0].axis('scaled')

            flag = np.where(all_combinations[:, 0] == id_visualize)[0]
            colors = results[flag][:, None]
            colors = np.concatenate((colors, np.zeros_like(colors), np.zeros_like(colors)), axis=1)
            sc2 = axes[1].scatter(
                all_points[0][all_combinations[flag][:, 1]][:, 0],
                all_points[0][all_combinations[flag][:, 1]][:, 1],
                s=1, c=results[flag] + 1e-12,
                norm=matplotlib.colors.LogNorm(),
                # vmin = 0, vmax = 1,
            )
            axes[1].set_xlim(-0.5, 0.5)
            axes[1].set_ylim(-0.5, 0.5)
            axes[1].tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False)
            axes[1].axis('scaled')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sc2, cax=cax, orientation='vertical')

            # plt.tight_layout()
            # plt.show(block=True)
            plt.savefig("output/{}.jpg".format(id_visualize), dpi=150)
            plt.close()

    labels = None
    # Spectral
    if True:
        similarity_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        similarity_matrix[all_combinations[:, 1], all_combinations[:, 0]] = results
        similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.transpose())
        similarity_matrix = 1 - similarity_matrix

        from sklearn.cluster import SpectralClustering
        # Apply Spectral Clustering
        clustering = SpectralClustering(n_clusters=20, affinity='precomputed', assign_labels='discretize')
        clustering.fit(similarity_matrix)
        labels = clustering.labels_

    # KMEANS
    if False:
        similarity_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        similarity_matrix[all_combinations[:, 1], all_combinations[:, 0]] = results
        similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.transpose())
        from sklearn.cluster import KMeans
        from sklearn.manifold import MDS
        embedding = MDS(n_components=2, dissimilarity='precomputed')
        X_transformed = embedding.fit_transform(similarity_matrix)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=20)
        kmeans.fit(X_transformed)
        labels = kmeans.labels_

    # Print the cluster labels
    plt.scatter(all_points[0][:, 0], all_points[0][:, 1], s=1, c=labels, cmap="tab10")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.axis('scaled')
    plt.colorbar()
    plt.show(block=True)

    level_set_points = find_level_set(query_points, udf, g, 0.05, 100)

    num_queries = query_points.shape[0]

    num_pair = all_combinations.shape[0]
    selected_points = query_points[all_combinations]
    selected_nearest_surface_points = nearest_surface_points[all_combinations]

    batch_size = 100000
    num_batch = int(np.ceil(num_pair / batch_size))
    result_similarities = []
    for i in tqdm(range(num_batch)):
        times = [0] * 10
        cur = time.time()
        id_start = batch_size * i
        id_end = batch_size * (i + 1)
        num_edge_points, sampled_edge_points = sample_edge(selected_nearest_surface_points[id_start:id_end])
        times[0] += time.time() - cur
        cur = time.time()
        udf = kdtree.search(sampled_edge_points, 1)[0][:, 0]
        times[1] += time.time() - cur
        cur = time.time()
        result = np.bincount(np.arange(num_edge_points.shape[0], dtype=np.int64).repeat(num_edge_points),
                             weights=udf)
        result = result / num_edge_points
        times[2] += time.time() - cur
        cur = time.time()
        result_similarities.append(result)

        if False:
            result = result.reshape(100, 100)
            plt.scatter(query_points[:, 0], query_points[:, 1], c=result)
            plt.scatter(target_vertices[:, 0], target_vertices[:, 1], c=(1, 0, 0), s=1)
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.colorbar()
            plt.show()
        pass

    pass


if __name__ == '__main__':
    print("Start to construct dataset")
    np.random.seed(0)

    generate_training_data()
