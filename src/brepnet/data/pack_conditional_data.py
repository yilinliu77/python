import shutil, os
from pathlib import Path
from tqdm import tqdm

data_train_root = Path(r"/mnt/d/img2brep/deepcad_train_v6")
data_validation_root = Path(r"/mnt/d/img2brep/deepcad_validation_v6")
data_test_root = Path(r"/mnt/d/img2brep/deepcad_test_v6")
latent_root = Path(r"/mnt/d/img2brep/ae_0925_7m_gaussian")
img_npz_root = Path(r"/mnt/d/img2brep/deepcad_730_imgs_npz")
img_png_root = Path(r"/mnt/d/img2brep/deepcad_730_imgs_png")
list_file = r"src/brepnet/data/list/deduplicated_deepcad_7_30.txt"

output_root = Path(r"/mnt/d/img2brep/deepcad_cond_v0")

if __name__ == "__main__":
    valid_folders = [item.strip() for item in open(list_file, "r").readlines()]
    valid_folders.sort()

    for idx, v_folder in enumerate(tqdm(valid_folders)):
        output_root_folder = output_root / v_folder
        output_root_folder.mkdir(parents=True, exist_ok=True)

        # Copy latent
        shutil.copy(latent_root / v_folder/ f"features.npy", output_root_folder / "features.npy")

        # Copy imgs
        shutil.copy(img_npz_root / v_folder / f"data.npz", output_root_folder / "imgs.npz")
        
        if (data_train_root/v_folder).exists():
            shutil.copyfile(data_train_root / v_folder / "pc.ply", output_root_folder / "pc.ply")
        elif (data_validation_root/v_folder).exists():
            shutil.copyfile(data_validation_root / v_folder / "pc.ply", output_root_folder / "pc.ply")
        elif (data_test_root/v_folder).exists():
            shutil.copyfile(data_test_root / v_folder / "pc.ply", output_root_folder / "pc.ply")
