import os, shutil
from tqdm import tqdm

data_root1 = r"/mnt/d/img2brep/0909_test"
data_root2 = r"/mnt/d/img2brep/deepcad_whole_test_v4"
save_root = r"/mnt/d/img2brep/0909_test_export"
os.makedirs(save_root, exist_ok=True)
all_files = os.listdir(data_root1)
all_files.sort()

folder_names = []
for filename in tqdm(all_files):
    if filename.endswith("_feature.npz"):
        folder_names = filename.split("_")[0]
        # option = ["_fixed", "_failed_-1", "invalid_3"]
        option = [""]
        src_folder = None
        for postfix in option:
            if os.path.exists(os.path.join(data_root2, folder_names + postfix)):
                src_folder = os.path.join(data_root2, folder_names + postfix)
                break
        if not src_folder:
            # print(f"{folder_names} No optional folder")
            continue
            # raise
        if not os.path.exists(os.path.join(save_root, folder_names)):
            shutil.copytree(src_folder, os.path.join(save_root, folder_names))

        if not os.path.exists(os.path.join(save_root, folder_names, filename)):
            shutil.copy(os.path.join(data_root1, filename), os.path.join(save_root, folder_names, filename))

        filename = filename.replace("_feature.npz", ".npz")
        if not os.path.exists(os.path.join(save_root, folder_names, filename)):
            shutil.copy(os.path.join(data_root1, filename), os.path.join(save_root, folder_names, filename))
