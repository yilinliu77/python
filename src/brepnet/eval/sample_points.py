import os
from pathlib import Path

import trimesh
import ray
from trimesh.sample import sample_surface

@ray.remote
def sample(prefix, root, output_root):
    file = root / prefix / "recon_brep.ply"
    if not file.exists():
        return
    mesh = trimesh.load(str(file), force="mesh")
    if mesh.faces.shape[0] == 0:
        return
    out_pc, _ = sample_surface(mesh, 2000)
    trimesh.PointCloud(out_pc).export(str(output_root / (prefix + ".ply")))
    pass

if __name__ == '__main__':
    ray.init(
        # num_cpus=1,
        # local_mode=True,
    )

    root = Path("C:/Users/yilin/Desktop/test_output_post1")
    output_root = Path("C:/Users/yilin/Desktop/test_output_points1")
    output_root.mkdir(exist_ok=True, parents=True)

    tasks = []
    for prefix in os.listdir(root):
        if not prefix.startswith("00"):
            continue
        tasks.append(sample.remote(prefix, root, output_root))
    ray.get(tasks)
    os.listdir()