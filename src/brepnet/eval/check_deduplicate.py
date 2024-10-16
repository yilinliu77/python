import networkx as nx
import numpy as np
import argparse
import os

from tqdm import tqdm
import ray


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


def compute_gen_novel():
    pass


def compute_gen_unique(graph_list, is_use_ray=False, batch_size=100000):
    N = len(graph_list)
    unique_graph_idx = list(range(N))
    pair_0, pair_1 = np.triu_indices(N, k=1)
    check_pairs = list(zip(pair_0, pair_1))

    if not is_use_ray:
        for idx1, idx2 in tqdm(check_pairs):
            is_identical = is_graph_identical(graph_list[idx1], graph_list[idx2])
            if is_identical:
                unique_graph_idx.remove(idx2) if idx2 in unique_graph_idx else None
    else:
        ray.init()
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
                unique_graph_idx.remove(idx2) if idx2 in unique_graph_idx else None

    unique = len(unique_graph_idx)
    print(f"Unique: {unique}/{N}")
    unique_ratio = unique / N
    return unique_ratio


def load_data_with_prefix(root_folder, prefix):
    data_files = []

    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files


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
    if 'sample_points_faces' in data_npz and 'edge_face_connectivity' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
        edge_points = data_npz['sample_points_lines']  # Edge sample points (num_lines*20*3)
        edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
        edge_points = data_npz['pred_edge']
        edge_face_connectivity = data_npz['pred_edge_face_connectivity']
    else:
        raise ValueError("Invalid data format")
    faces_adj_pair = []
    for edge_idx, face_idx1, face_idx2 in edge_face_connectivity:
        faces_adj_pair.append([face_idx1, face_idx2])
    return face_points, faces_adj_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str, required=True)
    parser.add_argument("--train", type=str, required=False)
    parser.add_argument("--n_bit", type=int, default=4)
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--batch_size", type=int, default=200000)
    args = parser.parse_args()
    gen_data_root = args.fake
    train_data_root = args.train
    is_use_ray = args.use_ray
    n_bit = args.n_bit
    batch_size = args.batch_size

    # Load all the generated data files
    print("Loading generated data files...")
    data_npz_file_list = load_data_with_prefix(gen_data_root, '.npz')
    # data_npz_file_list = data_npz_file_list[:100]
    gen_graph_list = []
    for data_npz_file in tqdm(data_npz_file_list):
        faces, faces_adj_pair = load_data_from_npz(data_npz_file)
        graph = build_graph(faces, faces_adj_pair, n_bit)
        gen_graph_list.append(graph)

    print("Computing unique ratio...")
    unique_ratio = compute_gen_unique(gen_graph_list, is_use_ray, batch_size)
    print(f"Unique ratio: {unique_ratio}")
    pass


if __name__ == "__main__":
    main()
