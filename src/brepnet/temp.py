import os
import shutil
from tqdm import tqdm

data_root_src = r"E:\data\img2brep\ourgen\deepcad_test_pcd"
data_root_our = r"E:\data\img2brep\ourgen\deepcad_v6\deepcad_test_v6"
out_root = r"E:\data\img2brep\ourgen\deepcad_test_pcd_our"
os.makedirs(out_root, exist_ok=True)

folders = os.listdir(data_root_our)

for folder in tqdm(folders):
    file_name = folder + ".ply"
    if os.path.exists(os.path.join(data_root_src, file_name)):
        shutil.copy(os.path.join(data_root_src, file_name), os.path.join(out_root, file_name))
