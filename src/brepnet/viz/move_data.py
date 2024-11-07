import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

post_root = Path("/mnt/d/cond_results/1025_sv_gaussian_epsilon_1400k_post")

if __name__ == "__main__":
    name = post_root.name.split("_post")[0]
    out_root = post_root.parent / (name+"_viz")
    out_root.mkdir(exist_ok=True)
    folders = [f for f in post_root.iterdir() if f.is_dir()]
    for folder in tqdm(folders):
        if not (folder / "success.txt").exists():
            continue
        shutil.copyfile(folder / "recon_brep.stl", out_root / (folder.name+".stl"))
        pngs = list((post_root.parent/name/folder.name).glob("*.png"))
        if len(pngs) == 4:
            pngs = [np.pad(cv2.imread(str(png)), ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=255) for png in pngs]
            total_png = cv2.vconcat((cv2.hconcat(pngs[:2]), cv2.hconcat(pngs[2:])))
            cv2.imwrite(str(out_root / (folder.name+".png")), total_png)
        elif len(pngs) == 1:    
            shutil.copyfile(pngs[0], out_root / (folder.name+f".png"))
        for condition in list((post_root.parent/name/folder.name).glob("*_pc.ply")):
            shutil.copyfile(condition, out_root / (folder.name+f"_pc.ply"))
        
