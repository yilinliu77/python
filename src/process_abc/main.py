from pathlib import Path
import ray
import os, shutil

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_SHAPE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer

root = Path("/mnt/e/data/")
output_root = Path("/mnt/e/ABC_STEP_SEPARATED")

def check_dir(v_path):
    if os.path.exists(v_path):
        shutil.rmtree(v_path)
    os.makedirs(v_path)

def explore_shape(shape, shape_type):
    explorer = TopExp_Explorer(shape, shape_type)
    results = []
    while explorer.More():
        results.append(explorer.Current())
        explorer.Next()
    return results

@ray.remote
def process_file(v_id):
    # Take the ".step" file
    step_file = [item for item in os.listdir(root / v_id) if item.endswith(".step")][0]
    step_file = str(root / v_id / step_file)
    # Read step file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)
    if status == 0:
        print(f"Error reading file {step_file}")
        return
    print(f"File {step_file} read successfully")
    num_shape = step_reader.TransferRoots()
    num_valid = 0
    for i in range(1, num_shape + 1):
        shape = step_reader.Shape(i)
        if shape.IsNull():
            continue
        faces = explore_shape(shape, TopAbs_FACE)
        if len(faces) < 7:
            continue

        out_path = str(output_root / "{:08d}_{:04d}.step".format(int(v_id), num_valid))
        # Write without logging
        step_writer = STEPControl_Writer()
        step_writer.Transfer(shape, STEPControl_AsIs)
        status = step_writer.Write(out_path)
        if status == 0:
            print(f"Error writing file {out_path}")
            continue
        num_valid+=1
    return

if __name__ == "__main__":
    ids = sorted([item for item in os.listdir(root)])

    if False:
        check_dir(output_root)
        for id in ids:
            process_file(id)
    else:
        ray.init(
            num_cpus=64,
            dashboard_host="0.0.0.0",
            dashboard_port="8999"
        )

        results = []
        for id in ids:
            results.append(process_file.remote(id))
        ray.get(results)
        print("Done")