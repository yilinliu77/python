from utils import *
import numpy as np

data_npz_path = r"D:\WorkSpace\BrepGen\samples_deepcad\6H4XQtsnGFylEbb_4\init_data.npz"


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


recon_face_points, recon_edge_points, face_edge_adj = get_data(data_npz_path)
recon_geom_faces = [create_surface(points) for points in recon_face_points]
recon_topo_faces = [BRepBuilderAPI_MakeFace(geom_face, 1e-3).Face() for geom_face in recon_geom_faces]
recon_curves = [create_edge(points) for points in recon_edge_points]
recon_edge = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in recon_curves]

trimmed_face_validity_list = []
connected_tolerance = 0.01
for face_idx, face_edge_adj_c in enumerate(face_edge_adj):
    geom_face = recon_geom_faces[face_idx]
    topo_face = recon_topo_faces[face_idx]
    face_edges = [recon_edge[i] for i in face_edge_adj_c]
    wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance)
    trimmed_face_validity_list.append(is_face_valid1)

print(trimmed_face_validity_list)
print(f"Total valid faces ratio: {sum(trimmed_face_validity_list) / len(trimmed_face_validity_list)}")
pass
