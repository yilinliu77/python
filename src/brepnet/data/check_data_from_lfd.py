import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from src.brepnet.viz.sort_and_merge import arrange_meshes
import ray


def process_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    local_folder_list = data[0]
    local_folder_list = [os.path.basename(item) for item in local_folder_list]
    local_removed_folder_list = []
    local_deduplicated_folder_pair_list = []
    local_unique_idx_list = list(range(len(local_folder_list)))
    local_lfd_matrix = data[1]
    idx0, idx1 = np.where(local_lfd_matrix < lfd_threshold)
    for i, j in list(zip(idx0, idx1)):
        if i < j:
            if j in local_unique_idx_list:
                local_unique_idx_list.remove(j)
                local_removed_folder_list.append(local_folder_list[j])
                local_deduplicated_folder_pair_list.append((local_folder_list[i], local_folder_list[j]))
    local_unique_folder_list = [local_folder_list[i] for i in local_unique_idx_list]
    print(f"Pkl file path: {os.path.basename(pkl_file)}, Local Unique ratio: {len(local_unique_folder_list) / len(local_folder_list)}")
    return local_folder_list, local_removed_folder_list, local_deduplicated_folder_pair_list


process_pkl_remote = ray.remote(process_pkl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--pkl_root', type=str, required=True)
    parser.add_argument('--lfd', type=float, required=False, default=10)
    args = parser.parse_args()
    pkl_root = args.pkl_root
    lfd_threshold = args.lfd
    out_txt_path = pkl_root + f"brepgen_train_novel_lfd_{lfd_threshold}.txt"

    all_pkl_files = os.listdir(pkl_root)
    all_pkl_files = all_pkl_files[0:3]
    folder_list = []
    unique_folder_list = []
    removed_folder_list = []
    deduplicated_folder_pair_list = []

    ray.init(local_mode=False)
    futures = []
    for pkl_file in tqdm(all_pkl_files):
        if not pkl_file.endswith(".pkl"):
            continue
        futures.append(process_pkl_remote.remote(os.path.join(pkl_root, pkl_file)))
    for future in tqdm(futures):
        local_folder_list, local_removed_folder_list, local_deduplicated_folder_pair_list = ray.get(future)
        folder_list.extend(local_folder_list)
        removed_folder_list.extend(local_removed_folder_list)
        deduplicated_folder_pair_list.extend(local_deduplicated_folder_pair_list)

    folder_list = list(set(folder_list))
    removed_folder_list = list(set(removed_folder_list))
    unique_folder_list = list(set(folder_list) - set(removed_folder_list))

    unique_folder_list.sort()
    print(f"Unique ratio: {len(unique_folder_list) / len(folder_list)}")

    # Sample deduplicated
    # existing_items = set(os.listdir("D:/brepnet/deepcad_v6"))
    # unique_folder_list = list(set(unique_folder_list) & existing_items)
    # unique_folder_list = np.random.choice(unique_folder_list, 2500, replace=False)
    # unique_folder_list = [os.path.join("D:/brepnet/deepcad_v6", item, "mesh.ply") for item in unique_folder_list]
    # arrange_meshes(unique_folder_list, "D:/brepnet/deepcad_v6/test.ply")

    deduplicated_txt_path = pkl_root + f"brepgen_train_deduplicated_lfd_{lfd_threshold}.txt"
    with open(deduplicated_txt_path, "w") as f:
        for folder_pair in deduplicated_folder_pair_list:
            f.write(f"{folder_pair[0]} {folder_pair[1]}\n")

    with open(out_txt_path, "w") as f:
        for folder in unique_folder_list:
            f.write(f"{folder}\n")
