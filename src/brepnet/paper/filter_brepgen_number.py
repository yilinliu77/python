from pathlib import Path
import os, shutil

import ray
from src.brepnet.eval.check_valid import *

deepcad_root = Path("/mnt/d/brepgen_results/deepcad")
output_root = Path("/mnt/d/brepgen_results/deepcad_filtered")
output_root_75 = Path("/mnt/d/brepgen_results/deepcad_75")
sta_total = np.load("/mnt/d/brepgen_results/deepcad/num_totals.npy")
sta_valids = np.load("/mnt/d/brepgen_results/deepcad/num_valids.npy")
output_root.mkdir(exist_ok=True, parents=True)
(output_root_75 / "suce").mkdir(exist_ok=True, parents=True)
(output_root_75 / "fail").mkdir(exist_ok=True, parents=True)

ray.init()
random.seed(0)

brepgen_root = Path("/mnt/d/brepgen_results/brepgen_v4")
output_root = Path("/mnt/d/brepgen_results/brepgen_filtered")
num_items = 0
for folder in tqdm(list(brepgen_root.iterdir())):
    if not folder.is_dir():
        continue
    num_faces = np.load(folder / "data_src.npz")["pred_face"].shape[0]
    if num_faces >= 7:
        shutil.copytree(folder, output_root / folder.name, dirs_exist_ok=True)
        num_items += 1

print(f"Filtered {num_items}/{len(list(brepgen_root.iterdir()))} items")