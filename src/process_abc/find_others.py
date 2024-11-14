from pathlib import Path
import ray
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from tqdm import tqdm
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve
            
from shared.occ_utils import disable_occ_log, get_primitives

root = Path("/mnt/e/yilin/data_step/")
output_root = Path("/mnt/e/yilin/data_flag/")
disable_occ_log()

@ray.remote
def process_file(v_id):
    try:
        if (output_root/(v_id+"_others")).exists():
            return True
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

        faces = get_primitives(shape, TopAbs_FACE)
        edges = get_primitives(shape, TopAbs_EDGE)

        for face in faces:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                                            GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface]:
                open(output_root/(v_id+"_others"), "w").close()
                return True
        for edge in edges:
            edge = BRepAdaptor_Curve(edge)
            if edge.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve]:
                open(output_root/(v_id+"_others"), "w").close()
                return True
        return False
    except Exception as e:
        print(e)
        open(output_root/(v_id+"_others"), "w").close()
        return True


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
            # local_mode=True,
            # num_cpus=1,
        )

        tasks = []
        for id in ids:
            tasks.append(process_file.remote(id))
        valid_ids = []
        for i in tqdm(range(num_tasks)):
            if ray.get(tasks[i]):
                valid_ids.append(ids[i])
        np.savetxt("src/process_abc/abc_others.txt", valid_ids, fmt="%s")
        print("Found {} others".format(len(valid_ids)))