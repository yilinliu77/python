from pathlib import Path
from src.brepnet.eval.check_valid import *
import random

deepcad_root = Path("/mnt/d/brepgen_results/deepcad")
brepgen_root = Path("/mnt/d/brepgen_results/brepgen_v4")
ours_root = Path("/mnt/d/uncond_results/1123/1118_730_800k_post/")

out_root = Path("/mnt/d/uncond_results/ours_75")

shapedir = []
for item in tqdm(ours_root.iterdir()):
    if (item/"recon_brep.step").exists():
        valid, shape = check_step_valid_soild(item/"recon_brep.step", return_shape=True)
        if valid:
            if len(get_primitives(shape, TopAbs_FACE)) > 7:
                shapedir.append(item)

print("Total valid shapes:", len(shapedir))
print("Randomly sample 75")

random.seed(0)
random.shuffle(shapedir)
shapedir = shapedir[:75]

out_root.mkdir(exist_ok=True, parents=True)
for item in shapedir:
    idx = item.name
    shutil.copy(item/"recon_brep.step", out_root/f"{idx}.step")