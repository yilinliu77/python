from pathlib import Path
import os, shutil
from src.brepnet.eval.check_valid import *

brepgen_root = Path("/mnt/d/brepgen_results/brepgen_filtered")
names = []
for folder in tqdm(list(brepgen_root.iterdir())):
    if not folder.is_dir():
        continue
    valid = check_step_valid_soild(folder / "recon_brep.step")
    if not valid:
        names.append(folder.name)    
print(names[:10])