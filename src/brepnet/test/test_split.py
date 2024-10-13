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
    files = [item.strip() for item in open("1.txt").readlines()]
    exist_list = [item.strip() for item in open("src/brepnet/data/list/deduplicated_abc_testing.txt").readlines()]
    print(len(exist_list))
    results = list(set(files) & set(exist_list))
    results.sort()
    print(len(results))
    np.savetxt("deduplicated_abc_testing_brepnet.txt", results, "%s")
