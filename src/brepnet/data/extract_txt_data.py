import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

root_path = Path(r"/mnt/d/img2brep/deepcad_730_cond")
csv_path = r"/mnt/d/img2brep/text2cad_v1.1.csv"

if __name__ == "__main__":
    # outputdir = Path(r"/mnt/d/img2brep/deepcad_730_cond")
    # outputdir.mkdir(exist_ok=True)
    # for item in tqdm(root_path.iterdir()):
    #     if item.is_dir():
    #         (outputdir/item.name).mkdir(exist_ok=True)
    #         if (item/"text_feat.npy").exists():
    #             shutil.copy(item/"text_feat.npy", outputdir/item.name)
    #             shutil.copy(item/"text.txt", outputdir/item.name)
    
    all_data = pd.read_csv(csv_path)
    for i in tqdm(range(all_data.shape[0])):
        data = all_data.loc[i]
        prefix = data["uid"][5:]
        if not (root_path/prefix).exists():
            continue
        abs = data["abstract"]
        beg = data["beginner"]
        with open(root_path/prefix/"text.txt", "w") as f:
            f.write(f"{abs}\n")
            f.write(f"{beg}\n")

    num_invalid = 0
    for prefix in tqdm(os.listdir(root_path)):
        if not (root_path/prefix/"text.txt").exists():
            num_invalid += 1
    print(num_invalid)
