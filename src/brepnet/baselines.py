from pathlib import Path

import numpy as np
import open3d as o3d
import ray
from tqdm import tqdm

testing_ids = [item.strip() for item in open(r"src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")]
root = Path(r"D:/brepnet/deepcad_v6/")
output_root = Path(r"D:/brepnet/baseline_pcs/")
(output_root / "poisson").mkdir(parents=True, exist_ok=True)
(output_root / "random").mkdir(parents=True, exist_ok=True)

ray.init()

@ray.remote
def sample_mesh(id, root, output_root):
    mesh = o3d.io.read_triangle_mesh(str(root / id / f"mesh.ply"))
    pc_sampled = mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=True)
    poisson_sampled = mesh.sample_points_poisson_disk(number_of_points=10000, use_triangle_normal=True)
    o3d.io.write_point_cloud(str(output_root / "poisson" / f"{id}_10000.ply"), poisson_sampled)
    o3d.io.write_point_cloud(str(output_root / "random" / f"{id}_10000.ply"), pc_sampled)

tasks = []
for id in testing_ids:
    tasks.append(sample_mesh.remote(id, root, output_root))
ray.get(tasks)