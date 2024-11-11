import shutil, os
from pathlib import Path
from tqdm import tqdm
import numpy as np

data_train_root = Path(r"/mnt/d/deepcad_730")
latent_root = Path(r"/mnt/d/deepcad_730_0925_gaussian_7m")
img_npz_root = Path(r"/mnt/d/deepcad_730_imgs_npz")
img_png_root = Path(r"/mnt/d/deepcad_730_imgs_png")
list_file = r"src/brepnet/data/list/deduplicated_deepcad_7_30.txt"

output_root = Path(r"/mnt/d/deepcad_730_v0")

if __name__ == "__main__":
    valid_folders = [item.strip() for item in open(list_file, "r").readlines()]
    valid_folders.sort()

    num=0
    valid_list = []
    for idx, v_folder in enumerate(tqdm(valid_folders)):
        try:
            output_root_folder = output_root / v_folder
            output_root_folder.mkdir(parents=True, exist_ok=True)

            # Copy latent
            shutil.copy(latent_root / v_folder/ f"features.npy", output_root_folder / "features.npy")

            # Copy imgs
            shutil.copy(img_npz_root / v_folder / f"data.npz", output_root_folder / "imgs.npz")
            
            # Copy pc
            shutil.copyfile(data_train_root / v_folder / "pc.ply", output_root_folder / "pc.ply")
            num += 1
            valid_list.append(v_folder)
        except:
            print(v_folder)
            shutil.rmtree(output_root_folder)
    print(num)
    np.savetxt("deduplicated_deepcad_7_30_final.txt", valid_list, "%s")