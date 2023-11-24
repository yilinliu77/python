import os,open3d,shutil
from pathlib import Path
import re, math
import ray
import torch
import trimesh

opj = os.path.join

# root = Path(r"G:\Dataset\ABC\raw_data\abc_0000_obj_v00")
root = r"/mnt/d/yilin/ABC_raw/data/"

@ray.remote
def process_item(v_files):
    flags=[True] * len(v_files)
    for idx, v_file in enumerate(v_files):
        files = os.listdir(opj(root, v_file))
        if len(files) <= 2:
            flags[idx]=False
            continue

        yml = [f for f in files if "features" in f][0]
        str = "".join(open(opj(root,v_file,yml)).readlines())
        if "BSpline" in str or "Other" in str or "Revolution" in str or "Extrusion" in str:
            flags[idx]=False
            continue

        types = [i.start() for i in re.finditer("type: ", str)]
        planes = [i.start() for i in re.finditer("type: Plane", str)]
        lines = [i.start() for i in re.finditer("type: Line", str)]
        if len(types) == 18 and len(planes) == 6 and len(lines) == 12:
            flags[idx]=False
            continue

        obj = [f for f in files if f.endswith(".obj")][0]
        mesh = trimesh.load_mesh(os.path.join(root, v_file, obj), process=False, maintain_order=True)
        if mesh.split().shape[0] != 1:
            flags[idx]=False
            continue

        # Get bounding box
        extent = mesh.bounding_box.extents
        max_ratio = 5
        if extent[0] / extent[1] > max_ratio or extent[1] / extent[0] > max_ratio:
            flags[idx]=False
            continue
        if extent[0] / extent[2] > max_ratio or extent[2] / extent[0] > max_ratio:
            flags[idx]=False
            continue
        if extent[1] / extent[2] > max_ratio or extent[2] / extent[1] > max_ratio:
            flags[idx]=False
            continue

        flags[idx]=True
    return flags

if __name__ == "__main__":
    files = os.listdir(root)
    files=sorted(files)

    ray.init()
    chunk_size = 1000
    num_chunks = math.ceil(len(files) / chunk_size)
    tasks = []
    for i in range(num_chunks):
        # process_item(files[:10])
        task = process_item.remote(files[i*chunk_size:min(len(files),(i+1)*chunk_size)])
        tasks.append(task)

    results = ray.get(tasks)
    results = sum(results,[])
    with open("ids.txt", "w") as f:
        for idx,task in enumerate(files):
            if results[idx]:
                f.write(task + "\n")
    print("Done")
