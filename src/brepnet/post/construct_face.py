from utils import *
import numpy as np
import argparse
from src.brepnet.eval.check_valid import load_data_with_prefix
import ray
from tqdm import tqdm


def get_data(v_filename):
    # specify the key to get the face points, edge points and edge_face_connectivity in data.npz
    # data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'), allow_pickle=True)['arr_0'].item()
    data_npz = np.load(v_filename, allow_pickle=True)
    if 'sample_points_faces' in data_npz and 'edge_face_connectivity' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
        edge_points = data_npz['sample_points_lines']  # Edge sample points (num_lines*20*3)
        edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
        face_edge_adj = None

    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
        edge_points = data_npz['pred_edge']
        edge_face_connectivity = data_npz['pred_edge_face_connectivity']
        face_edge_adj = None

    elif 'pred_face' in data_npz and 'face_edge_mask' in data_npz:
        face_points = data_npz['pred_face'].astype(np.float32)
        edge_points = data_npz['pred_edge'].astype(np.float32)
        face_edge_mask = data_npz['face_edge_mask']
        edge_idx = np.arange(edge_points.shape[0])
        edge_slice = [0] + list(np.cumsum(face_edge_mask.sum(-1)))
        face_edge_adj = []
        for i in range(face_points.shape[0]):
            face_edge_adj_c = edge_idx[edge_slice[i]:edge_slice[i + 1]]
            face_edge_adj.append(face_edge_adj_c)

    elif 'pred_face' in data_npz and 'face_edge_adj' in data_npz:
        face_points = data_npz['pred_face'].astype(np.float32)
        edge_points = data_npz['pred_edge'].astype(np.float32)
        face_edge_adj = data_npz['face_edge_adj']
        edge_face_connectivity = []
        N = face_points.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                intersection = list(set(face_edge_adj[i]).intersection(set(face_edge_adj[j])))
                if len(intersection) > 0:
                    edge_face_connectivity.append([intersection[0], i, j])
        edge_face_connectivity = np.array(edge_face_connectivity)
    else:
        raise ValueError(f"Unknown data npz format {v_filename}")

    return face_points, edge_points, face_edge_adj


def process_one(data_npz_path, connected_tolerance=0.1, is_log=False):
    recon_face_points, recon_edge_points, face_edge_adj = get_data(data_npz_path)
    recon_geom_faces = [create_surface(points) for points in recon_face_points]
    recon_curves = [create_edge(points) for points in recon_edge_points]
    recon_edge = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in recon_curves]

    trimmed_face_validity_list = []
    for face_idx, face_edge_adj_c in enumerate(face_edge_adj):
        geom_face = recon_geom_faces[face_idx]
        face_edges = [recon_edge[i] for i in face_edge_adj_c]
        wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance)
        trimmed_face_validity_list.append(is_face_valid1)

    if is_log:
        print(trimmed_face_validity_list)
        print(f"Total valid faces ratio: {sum(trimmed_face_validity_list) / len(trimmed_face_validity_list)}")

    return trimmed_face_validity_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--connect_tol", type=float, required=False, default=0.1)
    parser.add_argument("--use_ray", action='store_true')
    args = parser.parse_args()

    if args.prefix:
        process_one(os.path.join(args.data_root, args.connect_tol, args.prefix))
        exit(0)

    data_files = load_data_with_prefix(args.data_root, "init_data.npz")
    data_files.sort()
    data_files = data_files[0:100]
    print(f"Total data files: {len(data_files)}")

    if not args.use_ray:
        all_results = []
        pbar = tqdm(data_files)
        for data_npz_path in pbar:
            results = process_one(data_npz_path, args.connect_tol)
            all_results.extend(results)
            pbar.set_postfix({"valid_faces_ratio": sum(all_results) / len(all_results)})
        print(f"Total valid faces ratio: {sum(all_results) / len(all_results)}")
    else:
        ray.init()
        process_one_remote = ray.remote(process_one)
        futures = [process_one_remote.remote(data_npz_path, args.connect_tol) for data_npz_path in tqdm(data_files)]
        all_results = []
        for future in tqdm(futures):
            results = ray.get(future)
            all_results.extend(results)
        print(f"Total valid faces ratio: {sum(all_results) / len(all_results)}")
        ray.shutdown()


if __name__ == "__main__":
    main()
