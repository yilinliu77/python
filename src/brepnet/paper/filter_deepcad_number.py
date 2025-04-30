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

# Valid ratio: ignore the ones with less than 7 faces
# 75: 1) number of faces >= 7
@ray.remote
def process_75(idx):
    folder = deepcad_root / f"{idx:05d}"
    
    shape = None
    if (folder / "recon_brep.step").exists():
        valid, shape = check_step_valid_soild(folder / "recon_brep.step", return_shape=True)
        if shape is not None and len(get_primitives(shape, TopAbs_FACE)) < 7:
            return 0,0
    
    if sta_total[idx] == sta_valids[idx] and shape is not None and valid:
        return 1,1
    return 1,0

tasks = ray.get([process_75.remote(idx) for idx in tqdm(range(20000))])
num_complex = sum([task[0] for task in tasks])
num_valid = sum([task[1] for task in tasks])
candidates_75 = [idx for idx, task in enumerate(tasks) if task[0] == 1]

print(f"{num_valid}/{num_complex}")
print(f"{len(candidates_75)}")

random.shuffle(candidates_75)

num_stored = 0
for idx in tqdm(candidates_75):
    folder = deepcad_root / f"{idx:05d}"
    if not (folder / "recon_brep.step").exists():
        continue
    num_stored += 1
    valid = check_step_valid_soild(folder / "recon_brep.step")
    if valid and sta_valids[idx] == sta_total[idx]:
        shutil.copyfile(folder / "recon_brep.step", output_root_75 / "suce" / f"{folder.name}.step")
    else:
        shutil.copyfile(folder / "recon_brep.step", output_root_75 / "fail" / f"{folder.name}.step")
    if num_stored == 80:
        break

# 3000: 
@ray.remote
def process_3000(idx):
    folder = deepcad_root / f"{idx:05d}"
    if not folder.is_dir():
        return ""
    if sta_total[idx] != sta_valids[idx]:
        return ""
    if not (folder / "recon_brep.step").exists():
        return ""
    valid, shape = check_step_valid_soild(folder / "recon_brep.step", return_shape=True)
    if shape is None or not valid:
        return ""
    if len(get_primitives(get_primitives(shape, TopAbs_SOLID)[0], TopAbs_FACE)) < 7:
        return ""
    return f"{idx:05d}"

candidates_3000 = []
for idx in range(20000):
    candidates_3000.append(process_3000.remote(idx))
candidates_3000 = [item for item in tqdm(ray.get(candidates_3000)) if item != ""]
print(f"{len(candidates_3000)}")
random.shuffle(candidates_3000)
candidates_3000 = candidates_3000[:3000]
for idx in tqdm(candidates_3000):
    shutil.copytree(deepcad_root / idx, output_root / idx)