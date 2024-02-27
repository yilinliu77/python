from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
import open3d as o3d

root_dir=Path(r"G:/Projects/img2brep/data/00016422")

def bak():
    pcd = o3d.io.read_point_cloud(r"G:/Dataset/GSP/Paper/BSpline/pred_voronoi/00000003.ply")
    mesh = o3d.io.read_triangle_mesh(r"G:/Dataset/GSP/Paper/BSpline/mesh/00000003.ply")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    query_point = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_point)
    total_points = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(total_points[signed_distance.numpy() < 0])
    o3d.io.write_point_cloud(r"G:/Dataset/GSP/Paper/Teaser/src/internal_voronoi.ply", pcd)


if __name__ == '__main__':
    bak()
    lines_and_faces = yaml.load(open(root_dir/"features.yml"),yaml.CLoader)

    planes = []

    for line in lines_and_faces["curves"]:
        coords = np.asarray(line["location"])
    for plane in lines_and_faces["surfaces"]:
        params = np.asarray(plane["coefficients"])
        vertex_index = set(plane["vert_indices"])
        planes.append((params,vertex_index))

    # Recover the topologies
    adj = np.zeros((len(planes),len(planes)),dtype=np.int32)
    for id_plane1, plane1 in enumerate(tqdm(planes)):
        for id_plane2, plane2 in enumerate(planes):
            if id_plane1>=id_plane2:
                continue
            if len(plane1[1]&plane2[1])!=0:
                adj[id_plane1,id_plane2] = 1
                adj[id_plane2,id_plane1] = 1


    pass