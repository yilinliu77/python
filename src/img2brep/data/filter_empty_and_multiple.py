import os
from pathlib import Path

import ray
import numpy as np

from OCC.Extend.DataExchange import read_step_file

data_root = Path(r"/mnt/d/ABC_raw/data/")

# @ray.remote(num_cpus=1)
def get_brep(v_root, v_folders):
    single_loop_folder = []

    for idx, v_folder in enumerate(v_folders):
        all_files = os.listdir(v_root / v_folder)
        step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff]
        if len(step_file)==0:
            single_loop_folder.append(True)
            continue
        step_file = step_file[0]
        try:
            shape = read_step_file(str(v_root / v_folder / step_file), verbosity=False)
            if shape.NbChildren() != 1:
                single_loop_folder.append(True)
            else:
                single_loop_folder.append(False)
        except:
            single_loop_folder.append(True)

    return single_loop_folder


get_brep_ray = ray.remote(get_brep)

if __name__ == '__main__':
    total_ids = os.listdir(data_root)

    total_ids.sort()
    total_ids=total_ids[:500000]
    # single process
    if False:
        get_brep(data_root, total_ids)
    else:
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=15000,
            # num_cpus=1,
            # local_mode=True
        )
        batch_size = 1000
        num_batches = len(total_ids) // batch_size + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(
                get_brep_ray.remote(data_root,
                                    total_ids[i * batch_size:min(len(total_ids), (i + 1) * batch_size)]))
        flags = ray.get(tasks)
        print("Done")

    flags = [item for array in flags for item in array]
    with open("0513.txt", "w") as f:
        for flag, item in zip(flags,total_ids):
            if flag:
                f.write("%s\n" % item)
