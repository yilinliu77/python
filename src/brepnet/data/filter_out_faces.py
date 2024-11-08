import os
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

import os, numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

root_path = Path("/mnt/d/img2brep/brepgen_train")
input_list = Path("src/brepnet/data/list/deduplicated_deepcad_training_30.txt")

num_max = 30
num_min = 7

if __name__ == "__main__":
    prefies = [item.strip() for item in open(input_list).readlines()]
    print(len(prefies))
    results = []
    num_ratio_failed = 0
    num_min_face_failed = 0
    num_max_face_failed = 0
    num_unknown_failed = 0
    for prefix in tqdm(prefies):
        try:
            if not (Path("/mnt/d/yilin/img2brep/brepgen_ae_0925_7m_gaussian")/prefix).exists():
                print(1)

            num_faces = np.load(root_path / prefix / "data.npz")["sample_points_faces"].shape[0]
            if num_faces > num_max:
                num_max_face_failed += 1
                continue
            elif num_faces < num_min:
                num_min_face_failed += 1
                continue
            else:
                mesh = o3d.io.read_triangle_mesh(str(root_path/prefix/"mesh.ply"))
                v = np.asarray(mesh.vertices)
                dx = np.max(v[:, 0]) - np.min(v[:, 0])
                dy = np.max(v[:, 1]) - np.min(v[:, 1])
                dz = np.max(v[:, 2]) - np.min(v[:, 2])
                if dx / dy > 10 and dx / dz > 10 or dy / dx > 10 and dy / dz > 10 or dz / dx > 10 and dz / dy > 10:
                    num_ratio_failed += 1
                    continue
                results.append(prefix)
        except:
            num_unknown_failed += 1
    print(num_ratio_failed)
    print(num_min_face_failed)
    print(num_max_face_failed)
    print(num_unknown_failed)
    print(len(results))
    np.savetxt("deduplicated_deepcad_training_{}_{}.txt".format(num_min, num_max), results, "%s")
