from pathlib import Path
import os, shutil
from src.brepnet.eval.check_valid import *

# deepcad_root = Path("/mnt/d/brepgen_results/deepcad")
# output_root = Path("/mnt/d/brepgen_results/deepcad_filtered")
# num_items = 0
# for folder in tqdm(list(deepcad_root.iterdir())):
#     if not folder.is_dir():
#         continue
#     if (folder / "recon_brep.step").exists():
#         _, shape = check_step_valid_soild(folder / "recon_brep.step", return_shape=True)
#         if shape is None:
#             continue
#         if len(get_primitives(shape, TopAbs_FACE)) >= 7:
#             shutil.copytree(folder, output_root / folder.name, dirs_exist_ok=True)
#             num_items += 1
# print(f"Filtered {num_items}/{len(list(deepcad_root.iterdir()))} items")

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