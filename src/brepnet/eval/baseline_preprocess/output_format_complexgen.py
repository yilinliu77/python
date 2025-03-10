import argparse
import os
import shutil

from tqdm import tqdm

data_root = r"E:\data\img2brep\0rebuttle\test_obj_10"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--out_root", type=str, default=None)
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.out_root

    print(f"Data root: {data_root}")

    all_files = os.listdir(data_root)
    folder_names = []
    for file_name in tqdm(all_files):
        if not file_name.endswith(".json"):
            continue
        folder_name = file_name.split("_")[0]
        if folder_name in folder_names:
            continue
        folder_names.append(file_name.split("_")[0])
        local_folder_path = os.path.join(out_root, folder_name)
        os.makedirs(local_folder_path, exist_ok=True)

    for file_name in tqdm(all_files):
        folder_name = file_name.split("_")[0]
        if folder_name not in folder_names:
            continue
        shutil.copyfile(os.path.join(data_root, file_name), os.path.join(out_root, folder_name, file_name))
    print("Done")
