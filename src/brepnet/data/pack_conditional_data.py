import shutil, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import ray

data_train_root = Path(r"/mnt/d/img2brep/abc_v2")
img_npz_root = Path(r"/mnt/d/img2brep/abc_v2_npz")
img_png_root = Path(r"/mnt/d/img2brep/abc_v2_npz/")
txt_root = Path(r"/mnt/d/img2brep/deepcad_v6_txt")
list_file = r"src/brepnet/data/list/abc_total.txt"

output_root = Path(r"/mnt/d/img2brep/abc_v2_cond")

@ray.remote
def copy_folder(v_folder):
    if not (img_npz_root / v_folder / f"data.npz").exists():
        return None
    if not (img_npz_root / v_folder / f"img_feature_dinov2.npy").exists():
        return None
    if not (data_train_root / v_folder / f"pc.ply").exists():
        return None
    
    try:
        output_root_folder = output_root / v_folder
        output_root_folder.mkdir(parents=True, exist_ok=True)

        # Copy imgs
        shutil.copy(img_npz_root / v_folder / f"data.npz", output_root_folder / "imgs.npz")
        shutil.copy(img_npz_root / v_folder / f"img_feature_dinov2.npy", output_root_folder / "img_feature_dinov2.npy")
        
        # Copy pc
        shutil.copyfile(data_train_root / v_folder / "pc.ply", output_root_folder / "pc.ply")
        
        # txt
        # shutil.copyfile(txt_root / v_folder / "text.txt", output_root_folder / "text.txt")
        # shutil.copyfile(txt_root / v_folder / "text_feat.npy", output_root_folder / "text_feat.npy")
        
        return v_folder
    except:
        print(v_folder)
        shutil.rmtree(output_root_folder)
        return None

if __name__ == "__main__":
    valid_folders = [item.strip() for item in open(list_file, "r").readlines()]
    valid_folders.sort()

    ray.init()
    tasks = []
    for v_folder in valid_folders:
        tasks.append(copy_folder.remote(v_folder))

    valid_list = []
    for v_folder in tqdm(tasks):
        folder = ray.get(v_folder)
        if folder is not None:
            valid_list.append(folder)
    print(len(valid_list))
    np.savetxt("abc_brepnet.txt", valid_list, "%s")