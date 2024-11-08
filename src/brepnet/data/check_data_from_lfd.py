import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--pkl_root', type=str, required=True)
    parser.add_argument('--lfd', type=float, required=False, default=10)
    args = parser.parse_args()
    pkl_root = args.pkl_root
    lfd_threshold = args.lfd
    out_txt_path = os.path.join(pkl_root, f"brepgen_train_novel_lfd_{lfd_threshold}.txt")

    all_pkl_files = os.listdir(pkl_root)
    folder_list = []
    unique_folder_list = []
    for pkl_file in tqdm(all_pkl_files):
        if not pkl_file.endswith(".pkl"):
            continue
        print(f"Processing {pkl_file}")
        with open(os.path.join(pkl_root, pkl_file), "rb") as f:
            data = pickle.load(f)
        local_folder_list = data[0]
        local_unique_idx_list = list(range(len(local_folder_list)))
        local_lfd_matrix = data[1]
        idx0, idx1 = np.where(local_lfd_matrix < lfd_threshold)
        for i, j in tqdm(list(zip(idx0, idx1))):
            if i < j:
                if j in local_unique_idx_list:
                    local_unique_idx_list.remove(j)

        local_unique_folder_list = [local_folder_list[i] for i in local_unique_idx_list]
        print(f"Local Unique ratio: {len(local_unique_folder_list) / len(local_folder_list)}")
        folder_list.extend(local_folder_list)
        unique_folder_list.extend(local_unique_folder_list)

    unique_folder_list.sort()
    print(f"Unique ratio: {len(unique_folder_list) / len(folder_list)}")
    with open(out_txt_path, "w") as f:
        for folder in unique_folder_list:
            f.write(f"{folder}\n")
