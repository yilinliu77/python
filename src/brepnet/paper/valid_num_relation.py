from pathlib import Path
import os, shutil
from src.brepnet.eval.check_valid import *

# brepgen_root = Path("/mnt/d/brepgen_results/brepgen_filtered")
# num_faces = []
# valids = []
# for folder in tqdm(list(brepgen_root.iterdir())):
#     if not folder.is_dir():
#         continue
#     valid = check_step_valid_soild(folder / "recon_brep.step")
#     valids.append(valid)
#     num_faces.append(np.load(folder / "data_src.npz")["pred_face"].shape[0])

# print("{}/{}/{}".format(len(valids), len(num_faces), len(list(brepgen_root.iterdir()))))
# data = np.stack((np.array(num_faces), np.array(valids)), axis=1)
# np.save("/mnt/d/brepgen_results/brepgen_filtered_valid_relation.npy", data)

ours_root1 = Path("/mnt/d/uncond_results/1123/1118_730_800k/")
ours_root2 = Path("/mnt/d/uncond_results/1123/1118_730_800k_post/")
num_faces = []
valids = []
for folder in tqdm(list(ours_root2.iterdir())):
    if not folder.is_dir():
        continue
    valid = check_step_valid_soild(folder / "recon_brep.step")
    valids.append(valid)
    num_faces.append(np.load(ours_root1 / folder.name / "data.npz")["pred_face"].shape[0])

print("{}/{}/{}".format(len(valids), len(num_faces), len(list(ours_root1.iterdir()))))
data = np.stack((np.array(num_faces), np.array(valids)), axis=1)
np.save("/mnt/d/uncond_results/1123/1118_730_800k_valid_relation.npy", data)