import numpy as np
import trimesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Extend.DataExchange import read_step_file

from shared.occ_utils import normalize_shape, get_primitives, get_triangulations

if __name__ == '__main__':
    prefix = r"00107236"
    step_file = f"D:/Datasets/data_step/00107236/00107236_1516e8be2203f7a95a25b9e7_step_009.step"
    output_root = f"D:/Datasets/data_step/00107236/00107236_1516e8be2203f7a95a25b9e7_step_009.step"
    shape = read_step_file(step_file)
    shape = normalize_shape(shape)
    faces = get_primitives(shape, TopAbs_FACE)

    total_mesh = trimesh.Trimesh()
    for i in range(len(faces)):
        face = faces[i]
        v, f = get_triangulations(face)
        center = np.mean(v, axis=0)
        length = np.linalg.norm(center)
        dir = center / length
        v += dir * 0.5
        trimesh.Trimesh(vertices=v, faces=f).export(f"D:/brepnet/paper/teaser/face_{i}.ply")
        total_mesh += trimesh.Trimesh(vertices=v, faces=f)
    total_mesh.export(f"D:/brepnet/paper/teaser/{prefix}_total_mesh.ply")
    pass