import os
from pathlib import Path
import shutil

import ray
from tqdm import tqdm

v_root = Path("/mnt/e/data")
v_output = Path("/mnt/e/data_step")

@ray.remote
def process(prefix):
    id_start = prefix * 10000
    id_end =  prefix * 10000 + 10000
    for prefix in range(id_start, id_end):
        prefix = "{:08d}".format(prefix)
        all_files = os.listdir(v_root / prefix)
        step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff][0]
        (v_output/prefix).mkdir(exist_ok=True, parents=True)
        shutil.copyfile(v_root / prefix / step_file, v_output / prefix / step_file)

if __name__ == "__main__":
    prefixes = os.listdir(v_root)
    ray.init()

    tasks = []
    for prefix in range(100):
        tasks.append(process.remote(prefix))
    ray.get(tasks)