import os
import ray
import glob
from tqdm import tqdm
from OCC.Extend.DataExchange import read_step_file

data_root = r"/mnt/e/yilin/data_step"
all_folders = os.listdir(data_root)

def step2stl(step_folder):
    step_files = glob.glob(os.path.join(step_folder, "*.step"))
    if len(step_files) == 0:
        return
    step_file = step_files[0]
    stl_file = step_file.replace(".step", ".stl")
    shape = read_step_file(step_file)

