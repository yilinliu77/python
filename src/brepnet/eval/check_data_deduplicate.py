import multiprocessing

import networkx as nx
import numpy as np
import argparse
import os

import trimesh
from tqdm import tqdm
import ray

from src.brepnet.eval.check_valid import check_step_valid_soild, load_data_with_prefix
from src.brepnet.eval.eval_brepgen import normalize_pc


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2 ** n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)  # clip values
    return data_quantize.astype(int)


def build_graph(faces, faces_adj, n_bit=4):
    # faces1 and faces2 are np.array of shape (n_faces, n_points, n_points, 3)
    # faces_adj1 and faces_adj2 are lists of (face_idx, face_idx) adjacency, ex. [[0, 1], [1, 2]]
    faces_bits = real2bit(faces, n_bits=n_bit)
    """Build a graph from a shape."""
    G = nx.Graph()
    for face_idx, face_bit in enumerate(faces_bits):
        face_bit = face_bit.reshape(-1, 3)
        face_bit_ordered = face_bit[np.lexsort((face_bit[:, 0], face_bit[:, 1], face_bit[:, 2]))]
        G.add_node(face_idx, shape_geometry=face_bit_ordered)
    for pair in faces_adj:
        G.add_edge(pair[0], pair[1])
    return G


def is_graph_identical(graph1, graph2):
    """Check if two shapes are identical."""
    # Check if the two graphs are isomorphic considering node attributes
    return nx.is_isomorphic(
            graph1, graph2,
            node_match=lambda n1, n2: np.array_equal(n1['shape_geometry'], n2['shape_geometry'])
    )


def is_graph_identical_batch(graph_pair_list):
    is_identical_list = []
    for graph1, graph2 in graph_pair_list:
        is_identical = is_graph_identical(graph1, graph2)
        is_identical_list.append(is_identical)
    return is_identical_list


is_graph_identical_remote = ray.remote(is_graph_identical_batch)


def find_connected_components(matrix):
    N = len(matrix)
    visited = [False] * N
    components = []

    def dfs(idx, component):
        stack = [idx]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                for neighbor in range(N):
                    if matrix[node][neighbor] and not visited[neighbor]:
                        stack.append(neighbor)

    for i in range(N):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components


def generate_upper_triangle(N, batch_size):
    batch = []
    for i in range(N):
        for j in range(i + 1, N):
            if i == j:
                continue
            batch.append((i, j))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def compute_unique_bk(graph_list, is_use_ray=False, batch_size=100000):
    N = len(graph_list)
    identical_pairs = []
    unique_graph_idx = list(range(N))

    if not is_use_ray:
        check_pairs_yield = generate_upper_triangle(N, 1)
        pbar = tqdm(total=N * (N - 1) // 2, leave=False, dynamic_ncols=True)
        for batch_pairs in check_pairs_yield:
            idx1, idx2 = batch_pairs[0]
            pbar.update(1)
            if idx1 not in unique_graph_idx or idx2 not in unique_graph_idx:
                continue  # Skip if the graph is already removed
            is_identical = is_graph_identical(graph_list[idx1], graph_list[idx2])
            if is_identical:
                unique_graph_idx.remove(idx2) if idx2 in unique_graph_idx else None
            pbar.set_description(f"Unique: {len(unique_graph_idx)}/{N}")
    else:
        futures = []
        check_pairs_yield = generate_upper_triangle(N, batch_size)
        pbar = tqdm(total=(N * (N - 1) // 2) // batch_size, leave=False, dynamic_ncols=True)
        for batch_pairs in check_pairs_yield:
            batch_graph_pair = [(graph_list[idx1], graph_list[idx2]) for idx1, idx2 in batch_pairs]
            futures.append(is_graph_identical_remote.remote(batch_graph_pair))
            pbar.update(1)
        pbar.close()

        check_pairs_yield = generate_upper_triangle(N, batch_size)
        pbar = tqdm(total=(N * (N - 1) // 2) // batch_size)
        for batch_idx, batch_pairs in enumerate(check_pairs_yield):
            result = ray.get(futures[batch_idx])
            for idx, is_identical in enumerate(result):
                if not is_identical:
                    continue
                idx1, idx2 = batch_pairs[idx]
                if idx2 in unique_graph_idx:
                    unique_graph_idx.remove(idx2)
                identical_pairs.append((idx1, idx2))
            pbar.update(1)
        pbar.close()

    return unique_graph_idx, identical_pairs


def compute_unique(graph_list, is_use_ray=False, batch_size=100000, num_max_split_batch=128):
    N = len(graph_list)
    identical_pairs = []
    unique_graph_idx = list(range(N))
    pair_0, pair_1 = np.triu_indices(N, k=1)
    check_pairs = np.column_stack((pair_0, pair_1))

    num_split_batch = len(check_pairs) // batch_size
    if num_split_batch > 64:
        num_split_batch = num_max_split_batch
        batch_size = len(check_pairs) // num_split_batch

    if not is_use_ray:
        for idx1, idx2 in tqdm(check_pairs):
            is_identical = is_graph_identical(graph_list[idx1], graph_list[idx2])
            if is_identical:
                unique_graph_idx.remove(idx2) if idx2 in unique_graph_idx else None
    else:
        N_batch = len(check_pairs) // batch_size
        futures = []
        for i in tqdm(range(N_batch)):
            batch_pairs = check_pairs[i * batch_size: (i + 1) * batch_size]
            batch_graph_pair = [(graph_list[idx1], graph_list[idx2]) for idx1, idx2 in batch_pairs]
            futures.append(is_graph_identical_remote.remote(batch_graph_pair))
        results = ray.get(futures)

        for batch_idx in tqdm(range(N_batch)):
            for idx, is_identical in enumerate(results[batch_idx]):
                if not is_identical:
                    continue
                idx1, idx2 = check_pairs[batch_idx * batch_size + idx]
                if idx2 in unique_graph_idx:
                    unique_graph_idx.remove(idx2)
                identical_pairs.append((idx1, idx2))

    return unique_graph_idx, identical_pairs


def is_graph_identical_list(graph1, graph2_path_list):
    """Check if two shapes are identical."""
    # Check if the two graphs are isomorphic considering node attributes
    graph2_list, graph2_prefix_list = load_and_build_graph(graph2_path_list)
    for graph2 in graph2_list:
        if nx.is_isomorphic(graph1, graph2,
                            node_match=lambda n1, n2: np.array_equal(n1['shape_geometry'], n2['shape_geometry'])):
            return True
    return False


is_graph_identical_list_remote = ray.remote(is_graph_identical_list)


def test_check():
    sample = np.random.rand(3, 32, 32, 3)
    face1 = sample[[0, 1, 2]]
    face2 = sample[[0, 2, 1]]
    faces_adj1 = [[0, 1]]
    faces_adj2 = [[0, 2]]

    graph1 = build_graph(face1, faces_adj1)
    graph2 = build_graph(face2, faces_adj2)

    is_identical = is_graph_identical(graph1, graph2)
    # 判断图是否相等
    print("Graphs are equal" if is_identical else "Graphs are not equal")


def load_data_from_npz(data_npz_file):
    data_npz = np.load(data_npz_file, allow_pickle=True)
    data_npz1 = np.load(data_npz_file.replace("deepcad_32", "deepcad_train_v6"), allow_pickle=True)
    # Brepgen
    if 'face_edge_adj' in data_npz:
        faces = data_npz['pred_face']
        face_edge_adj = data_npz['face_edge_adj']
        faces_adj_pair = []
        N = face_edge_adj.shape[0]
        for face_idx1 in range(N):
            for face_idx2 in range(face_idx1 + 1, N):
                face_edges1 = face_edge_adj[face_idx1]
                face_edges2 = face_edge_adj[face_idx2]
                if sorted((face_idx1, face_idx2)) in faces_adj_pair:
                    continue
                if len(set(face_edges1).intersection(set(face_edges2))) > 0:
                    faces_adj_pair.append(sorted((face_idx1, face_idx2)))
        return faces, faces_adj_pair
    # Ours
    if 'sample_points_faces' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
        edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
        edge_face_connectivity = data_npz['pred_edge_face_connectivity']
    else:
        raise ValueError("Invalid data format")
    faces_adj_pair = []
    for edge_idx, face_idx1, face_idx2 in edge_face_connectivity:
        faces_adj_pair.append([face_idx1, face_idx2])
    if face_points.shape[-1] != 3:
        face_points = face_points[..., :3]

    src_shape = face_points.shape
    face_points = normalize_pc(face_points.reshape(-1, 3)).reshape(src_shape)
    return face_points, faces_adj_pair


def load_and_build_graph(data_npz_file_list, check_folders=None, n_bit=4):
    graph_list = []
    prefix_list = []
    for data_npz_file in data_npz_file_list:
        folder_name = os.path.basename(os.path.dirname(data_npz_file))
        if check_folders and folder_name not in check_folders:
            continue
        prefix_list.append(folder_name)
        faces, faces_adj_pair = load_data_from_npz(data_npz_file)
        graph = build_graph(faces, faces_adj_pair, n_bit)
        graph_list.append(graph)
    return graph_list, prefix_list


load_and_build_graph_remote = ray.remote(load_and_build_graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--n_bit", type=int, default=6)
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--load_batch_size", type=int, default=100)
    parser.add_argument("--compute_batch_size", type=int, default=10000)
    parser.add_argument("--txt", type=str, default=None)
    parser.add_argument("--num_cpus", type=int, default=32)
    args = parser.parse_args()
    train_data_root = args.train_root
    is_use_ray = args.use_ray
    n_bit = args.n_bit
    load_batch_size = args.load_batch_size
    compute_batch_size = args.compute_batch_size
    folder_list_txt = args.txt
    num_cpus = args.num_cpus

    if folder_list_txt:
        with open(folder_list_txt, "r") as f:
            check_folders = [line.strip() for line in f.readlines()]
    else:
        check_folders = None

    ################################################## Unqiue #######################################################
    # Load all the data files
    print("Loading data files...")
    data_npz_file_list = load_data_with_prefix(train_data_root, 'data.npz')
    data_npz_file_list.sort()
    if is_use_ray:
        ray.init()
        futures = []
        graph_list = []
        prefix_list = []
        for i in tqdm(range(0, len(data_npz_file_list), load_batch_size)):
            batch_data_npz_file_list = data_npz_file_list[i: i + load_batch_size]
            futures.append(load_and_build_graph_remote.remote(batch_data_npz_file_list, check_folders, n_bit))
        for future in tqdm(futures):
            result = ray.get(future)
            graph_list_batch, prefix_list_batch = result
            graph_list.extend(graph_list_batch)
            prefix_list.extend(prefix_list_batch)
        ray.shutdown()
    else:
        graph_list, prefix_list = load_and_build_graph(data_npz_file_list, n_bit)
    print(f"Loaded {len(graph_list)} data files")

    # sort the graph list according the face num
    graph_node_num = np.array([graph.number_of_nodes() for graph in graph_list])

    identical_pairs_txt = train_data_root + f"_identical_pairs_{n_bit}bit.txt"
    fp_identical_pairs = open(identical_pairs_txt, "w")
    fp_identical_pairs.close()
    novel_txt = train_data_root + f"_novel_{n_bit}bit.txt"
    fp_novel = open(novel_txt, "w")
    fp_novel.close()

    if is_use_ray:
        ray.init(_temp_dir=r"/mnt/d/img2brep/ray_temp")
    unique_graph_idx_list = []
    pbar = tqdm(range(3, 31))
    for num_face in pbar:
        if num_face != 6:
            continue
        print(f"Processing {num_face}")
        pbar.set_description(f"Processing {num_face}")
        fp_identical_pairs = open(identical_pairs_txt, "a")
        fp_novel = open(novel_txt, "a")
        print(f"face_num = {num_face}", file=fp_identical_pairs)

        hits_graph_idx = np.where(graph_node_num == num_face)[0]
        hits_graph = [graph_list[idx] for idx in tqdm(hits_graph_idx)]
        hits_graph_prefix = [prefix_list[idx] for idx in hits_graph_idx]

        if len(hits_graph) != 0:
            local_unique_graph_idx_list, identical_pairs = compute_unique(hits_graph, is_use_ray, compute_batch_size)
            for unique_graph_idx in local_unique_graph_idx_list:
                print(f"{hits_graph_prefix[unique_graph_idx]}", file=fp_novel)

            local_unique_graph_idx_list = [hits_graph_idx[idx] for idx in local_unique_graph_idx_list]
            unique_graph_idx_list.extend(local_unique_graph_idx_list)

            if len(identical_pairs) > 0:
                for idx1, idx2 in identical_pairs:
                    print(f"{hits_graph_prefix[idx1]} {hits_graph_prefix[idx2]}", file=fp_identical_pairs)
            pbar.set_postfix({"Local Unique": len(local_unique_graph_idx_list) / len(hits_graph),
                              "Total Unique": len(unique_graph_idx_list) / len(graph_list), })
            print(f"Unique: {len(local_unique_graph_idx_list)}/{len(hits_graph_idx)}"
                  f"={len(local_unique_graph_idx_list) / len(hits_graph_idx)}", file=fp_identical_pairs)
        else:
            print(f"face_num = {num_face} has no data", file=fp_identical_pairs)
        fp_identical_pairs.close()
        fp_novel.close()

    if is_use_ray:
        ray.shutdown()

    print(f"Unique num: {len(unique_graph_idx_list)}/{len(graph_list)}={len(unique_graph_idx_list) / len(graph_list)}")
    print(f"Identical pairs are saved to {identical_pairs_txt}")
    print(f"Novel txt are saved to {novel_txt}")
    print("Done")


if __name__ == "__main__":
    main()
