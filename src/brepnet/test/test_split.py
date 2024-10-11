import os
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

if __name__ == '__main__':
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