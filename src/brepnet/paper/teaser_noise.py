import numpy as np
import trimesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Extend.DataExchange import read_step_file

from shared.occ_utils import normalize_shape, get_primitives, get_triangulations

if __name__ == '__main__':
    pos = np.random.randn(1000,3)
    total_mesh = trimesh.Trimesh()
    for i in range(len(pos)):
        item = trimesh.primitives.Sphere(radius=0.02, subdivisions=1)
        item.apply_translation(pos[i])
        total_mesh += item
    total_mesh.export(f"D:/brepnet/paper/teaser/noise.ply")
    pass