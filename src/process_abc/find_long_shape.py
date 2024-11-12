from pathlib import Path
import ray
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.BRepTools import BRep_Tool
from OCC.Core import BRepBndLib, TopoDS
from OCC.Core.Bnd import Bnd_Box

from shared.occ_utils import diable_occ_log

ratio = 10

root = Path("/mnt/e/data/")
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
        assert status != 0
        num_shape = step_reader.TransferRoots()
        assert num_shape == 1
        shape = step_reader.Shape(1)
        assert shape.ShapeType() == TopAbs_SOLID

        boundingBox = Bnd_Box()
        BRepBndLib.brepbndlib.Add(shape, boundingBox)
        xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        if dx / dy > ratio and dx / dz > ratio or dy / dx > ratio and dy / dz > ratio or dz / dx > ratio and dz / dy > ratio:
            return True
        return False

    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    ids = sorted([item for item in os.listdir(root)])
    if not os.path.exists("src/process_abc/abc_single_solid.txt"):
        print("Run find_multiple_component.py first")
        exit()
    valid_ids = [item.strip() for item in open("src/process_abc/abc_single_solid.txt", "r").readlines()]
    ids = list(set(ids) & set(valid_ids))
    ids.sort()

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
        np.savetxt("src/process_abc/abc_long_shape.txt", valid_ids, fmt="%s")
        print("Found {} cubes".format(len(valid_ids)))