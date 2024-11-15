from pathlib import Path
import ray
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID
from tqdm import tqdm

from shared.occ_utils import disable_occ_log

root = Path("/mnt/e/yilin/data_step/")
output_root = Path("/mnt/e/yilin/data_flag/")
disable_occ_log()

@ray.remote(max_retries=0)
def process_file(v_id):
    if (output_root/(v_id+"_single_solid")).exists():
        return True
    try:
        # Take the ".step" file
        step_file = [item for item in os.listdir(root / v_id) if item.endswith(".step")][0]
        step_file = str(root / v_id / step_file)
        # Read step file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file)
        if status == 0:
            return False
        num_shape = step_reader.TransferRoots()
        if num_shape != 1:
            return False
        shape = step_reader.Shape(1)
        if shape.ShapeType() != TopAbs_SOLID:
            return False
        open(output_root/(v_id+"_single_solid"), "w").close()
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    ids = sorted([item for item in os.listdir(root)])
    # ids=ids[:1000]
    num_tasks = len(ids)
    print("Total files: ", num_tasks)
    output_root.mkdir(exist_ok=True)

    if True:
        valid_ids = []
        for id in tqdm(os.listdir(output_root)):
            id = id[:8]
            valid_ids.append(id)
        valid_ids.sort()
        np.savetxt("src/process_abc/abc_single_solid.txt", valid_ids, fmt="%s")
        print("Found {} valid ids".format(len(valid_ids)))
    else:
        ray.init(
        )

        tasks = []
        for id in ids:
            tasks.append(process_file.remote(id))
        valid_ids = []
        for i in range(num_tasks):
            if ray.get(tasks[i]):
                valid_ids.append(ids[i])
        np.savetxt("src/process_abc/abc_single_solid.txt", valid_ids, fmt="%s")
        print("Found {} valid ids".format(len(valid_ids)))