import os
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

import os, numpy as np
from pathlib import Path

from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import open3d as o3d


if __name__ == '__main__':
    root = Path("D:/brepnet/deepcad_v6_cond")
    out_root = Path("D:/brepnet/deepcad_v6_mvr_test")
    out_root.mkdir(exist_ok=True)

    all_folders = os.listdir(root)

    for folder in all_folders[:20]:
        imgs = np.load(root / folder / "imgs.npz")
        imgs = imgs["mvr_imgs"]

        r = np.random.randint(0, 63)
        idx = np.random.choice(np.arange(8), 4, replace=False)
        img = imgs[idx * 64 + r]
        for i, item in enumerate(img):
            Image.fromarray(item).save(out_root / f"{folder}_{idx[i]}.png")

        pc = np.asarray(o3d.io.read_point_cloud(str(root / folder / "pc.ply")).points)
        id_aug = r
        angles = np.array([
            id_aug % 4,
            id_aug // 4 % 4,
            id_aug // 16
        ])
        matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix()
        pc = (matrix @ pc.T).T

        o3d.io.write_point_cloud(str(out_root / f"{folder}_{r}_pc.ply"), o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc)))

    pass