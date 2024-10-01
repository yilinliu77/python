from pathlib import Path

import gurobipy as gp
import numpy as np
import trimesh

output_root = Path(r"D:/brepnet/debug_dir")

edge_colors = np.asarray([
    [1.0, 0.0, 0, 1],
    [0.9, 0.1, 0, 1],
    [0.8, 0.2, 0, 1],
    [0.7, 0.3, 0, 1],
    [0.6, 0.4, 0, 1],
    [0.5, 0.5, 0, 1],
    [0.4, 0.6, 0, 1],
    [0.3, 0.7, 0, 1],
    [0.2, 0.8, 0, 1],
    [0.1, 0.9, 0, 1],
    [0.1, 1.0, 0, 1],
    [0.1, 1.0, 0, 1],
    [0.1, 1.0, 0, 1],
    [0.1, 1.0, 0, 1],
    [0.1, 1.0, 0, 1],
    [0.1, 1.0, 0, 1],
])

if __name__ == '__main__':
    data = np.load(r"C:/Users/yilin/Documents/WeChat Files/what_seven/FileStorage/File/2024-09/00000093.npz")
    face_points = data['pred_face']
    edge_points = data['pred_edge']
    edge_face_connectivity = data['pred_edge_face_connectivity']

    num_faces = face_points.shape[0]
    num_edges = edge_points.shape[0]

    face_adj = -np.ones((num_faces, num_faces), dtype=np.int64)
    face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = edge_face_connectivity[:, 0]

    # Debug code
    output_root.mkdir(exist_ok=True, parents=True)
    for i_face in range(num_faces):
        points = []
        colors = []
        f = face_points[i_face].reshape(-1, 3)
        points.append(f)
        colors.append(np.array([1, 1, 0, 1])[None, :].repeat(f.shape[0], axis=0))
        for i_face2 in range(num_faces):
            if i_face == i_face2:
                continue
            if face_adj[i_face, i_face2] >= 0:
                edge = edge_points[face_adj[i_face, i_face2]]
                points.append(edge)
                colors.append(edge_colors)
        points = np.concatenate(points, axis=0)
        colors = (np.concatenate(colors, axis=0) * 255).astype(np.uint8)
        trimesh.PointCloud(points, colors).export(output_root / f"{i_face}.ply")



    pass
