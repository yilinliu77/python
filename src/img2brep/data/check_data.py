from pathlib import Path

import numpy as np
import open3d as o3d

from shared.common_utils import *

if __name__ == '__main__':
    root = Path(r"G:/Dataset/img2brep/deepcad_test")
    viz_id = "00000007"

    mesh = o3d.io.read_triangle_mesh(str(root/viz_id/"mesh.ply"))

    data = np.load(root/viz_id/"data.npz")

    face_points = data["sample_points_faces"]
    edge_points = data["sample_points_lines"]
    vertex_points = data["sample_points_vertices"]

    edge_face_connectivity = data["edge_face_connectivity"]
    vertex_edge_connectivity = data["vertex_edge_connectivity"]

    print("Show mesh")
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    print("Show edge face connectivity")
    for id_edge, id_face1, id_face2 in edge_face_connectivity:

        edge = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_points[id_edge])).paint_uniform_color([1, 0, 0])

        face = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.concatenate([face_points[id_face1].reshape(-1,3),face_points[id_face2].reshape(-1,3)],axis=0)
        )).paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([mesh,face,edge], mesh_show_wireframe=True)

    pass