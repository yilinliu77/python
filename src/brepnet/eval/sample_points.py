import os
from pathlib import Path

import trimesh
import ray
from trimesh.sample import sample_surface

import argparse

import glob


@ray.remote
def sample(prefix, root, output_root):
    option_file = glob.glob(os.path.join(root / prefix, '*.ply')) + glob.glob(os.path.join(root / prefix, '*.stl'))
    if len(option_file) == 0:
        return
    file = option_file[0]
    mesh = trimesh.load(str(file), force="mesh")
    if mesh.faces.shape[0] == 0:
        return
    out_pc, _ = sample_surface(mesh, 2000)
    trimesh.PointCloud(out_pc).export(str(output_root / (prefix + ".ply")))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, default=r"E:\data\img2brep\ourgen\uncond_gaussian_450k_post")
    parser.add_argument('--out_root', type=str, default=r"E:\data\img2brep\ourgen\uncond_gaussian_450k_post_sampled_pc")

    root = Path(parser.parse_args().data_root)
    output_root = Path(parser.parse_args().out_root)
    output_root.mkdir(exist_ok=True, parents=True)

    ray.init(
            # num_cpus=1,
            # local_mode=True,
    )

    tasks = []
    for prefix in os.listdir(root):
        tasks.append(sample.remote(prefix, root, output_root))
    ray.get(tasks)
    os.listdir()
