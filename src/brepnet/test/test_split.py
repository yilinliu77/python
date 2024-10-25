import os
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

import os, numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

root = Path("/mnt/d/abc_v0")

num_max_faces = 64

if __name__ == '__main__2':
    root = Path(r"D:/Datasets/data_step")
    lists = [item.strip() for item in open(
        r"C:\repo\python\src\brepnet\data\list\deduplicated_deepcad_training.txt").readlines()]
    num_faces = []
    for prefix in lists[:10000]:
        if not (root / prefix).exists():
            continue
        filename = os.listdir(root / prefix)[0]
        shp = read_step_file(str(root / prefix / filename))
        exp = TopExp_Explorer(shp, TopAbs_ShapeEnum.TopAbs_FACE)
        faces = []
        while exp.More():
            faces.append(exp.Current())
            exp.Next()
        if len(faces) == 6:
            print(prefix)
        num_faces.append(len(faces))
    print(num_faces)

if __name__ == "__main__1":
    valid_names = []
    for folder in tqdm(root.glob("*")):
        if (folder / "data.npz").exists():
            num_faces = np.load(str(folder / "data.npz"))["sample_points_faces"].shape[0]
            if num_faces < num_max_faces:
                valid_names.append(folder.name)
    valid_names.sort()
    np.savetxt("1.txt", valid_names, "%s")
    
if __name__ == "__main__":
    num_max = 30
    num_min = 7
    prefies = [item.strip() for item in open("src/brepnet/data/list/deduplicated_deepcad_training.txt").readlines()]
    zdataset = os.listdir("D:/brepnet/ae_0925_7m/ae_0925_7m_gaussian")
    root_path = Path("D:/brepnet/deepcad_train_v6")
    print(len(prefies))
    print(len(zdataset))
    valid_sets = set(prefies) & set(zdataset)
    # valid_sets = prefies
    results = []
    num_ratio_failed = 0
    num_min_face_failed = 0
    num_max_face_failed = 0
    num_unknown_failed = 0
    for prefix in tqdm(valid_sets):
        try:
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
                    num_ratio_failed +=1
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
