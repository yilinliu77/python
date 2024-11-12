from pathlib import Path
import ray
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID

from shared.occ_utils import diable_occ_log

root = Path("/mnt/e/yilin/data/")
diable_occ_log()

@ray.remote
def process_file(v_id):
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
        if num_shape == 1:
            return False
        shape = step_reader.Shape(1)
        if shape.ShapeType() != TopAbs_SOLID:
            return False
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    ids = sorted([item for item in os.listdir(root)])
    num_tasks = len(ids)
    print("Total files: ", num_tasks)

    if False:
        check_dir(output_root)
        for id in ids:
            process_file(id)
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