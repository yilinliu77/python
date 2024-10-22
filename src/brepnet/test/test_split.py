import os
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

import os, numpy as np
from pathlib import Path
from tqdm import tqdm

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
    prefies = [item.strip() for item in open("src/brepnet/data/list/deduplicated_deepcad_testing.txt").readlines()]
    zdataset = os.listdir("/mnt/d/img2brep/ae_0925_7m_gaussian")
    print(len(prefies))
    print(len(zdataset))
    # valid_sets = set(prefies) & set(zdataset)
    valid_sets = prefies
    results = []
    for prefix in tqdm(valid_sets):
        try:
            num_faces = np.load("/mnt/d/img2brep/deepcad_test_v6/" + prefix + "/data.npz")["sample_points_faces"].shape[0]
            if num_faces < num_max:
                results.append(prefix)
        except:
            continue
    print(len(results))
    np.savetxt("deduplicated_deepcad_testing_{}.txt".format(num_max), results, "%s")
