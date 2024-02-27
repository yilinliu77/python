import os
from pathlib import Path

import ray
import trimesh
import yaml
from tqdm import tqdm

data_root = Path(r"G:/Dataset/ABC/raw_data/abc_0000_obj_v00")
max_ratio = 5

@ray.remote(num_cpus=1)
def filter_mesh(v_root, v_folders, max_ratio):
    valid_ids = [True] * len(v_folders)
    for idx, v_folder in enumerate(v_folders):
        for ff in os.listdir(v_root / v_folder):
            if not ff.endswith(".obj"):
                continue
            mesh = trimesh.load_mesh(v_root / v_folder / ff, process=False, maintain_order=True)
            if mesh.faces.shape[0] < 10 or mesh.vertices.shape[0] < 10:
                valid_ids[idx]=False
                continue

            if mesh.split().shape[0] != 1:
                valid_ids[idx] = False
                continue

            # Get bounding box
            extent = mesh.bounding_box.extents
            if extent[0] / extent[1] > max_ratio or extent[1] / extent[0] > max_ratio:
                valid_ids[idx] = False
                continue
            if extent[0] / extent[2] > max_ratio or extent[2] / extent[0] > max_ratio:
                valid_ids[idx] = False
                continue
            if extent[1] / extent[2] > max_ratio or extent[2] / extent[1] > max_ratio:
                valid_ids[idx] = False
                continue

    return valid_ids

# Test all files under data_root
# - Filter out empty meshes
# - Filter out non-planar meshes
# - Filter out cube meshes
def get_splits(data_root):
    valid_ids = []
    cube_ids = []

    for folder in tqdm(os.listdir(data_root)):
        for ff in os.listdir(data_root / folder):
            if ff.endswith(".yml") and "features" in ff:
                # file = yaml.load(
                #     open(os.path.join(r"G:/Dataset/ABC/raw_data/abc_0000_obj_v00", folder, ff), "r"),
                # Loader=yaml.CLoader)
                # is_pure_plane = True
                # for item in file["surfaces"]:
                #     if "plane" != item["type"]:
                #         is_pure_plane=False
                #         break
                # if is_pure_plane:
                #     print(folder)

                is_pure_plane = True
                str = open(os.path.join(r"G:/Dataset/ABC/raw_data/abc_0000_obj_v00", folder, ff), "r").read()
                pos = -1
                num_plane = 0
                num_line = 0
                while True:
                    pos = str.find("type:", pos + 1)
                    if pos == -1:
                        break
                    if str[pos + 6:pos + 8] not in ["Li", "Pl"]:
                        is_pure_plane = False
                        break
                    if str[pos + 6:pos + 8] == "Pl":
                        num_plane += 1
                    else:
                        num_line += 1

                if not is_pure_plane:
                    continue

                if num_plane == 6 and num_line == 12:
                    cube_ids.append(folder)

                valid_ids.append(folder)

    with open("planar_shapes_id.txt", "w") as f:
        for item in valid_ids:
            f.write(item + "\n")

    with open("cube_ids.txt", "w") as f:
        for item in cube_ids:
            f.write(item + "\n")

    with open("planar_shapes_except_cube.txt", "w") as f:
        new_set = set(valid_ids) - set(cube_ids)
        for item in new_set:
            f.write(item + "\n")

# Test all files under data_root
# - Filter out small shapes
# - Filter out meshes that contain multiple parts
# - Filter out meshes that have unreasonable ratio
def test_obj(data_root, split_file):
    # Test connectivity
    ray.init()

    ids = sorted([item.strip() for item in open(split_file).readlines()])

    num_batches = 80 # Larger than number of cpus for a more efficient task assignment
    batch_size = len(ids) // num_batches

    tasks = []
    for idx in range(num_batches + 1):
        task = filter_mesh.remote(data_root, ids[batch_size*idx:min(len(ids), batch_size*(idx+1))], max_ratio)
        # result = ray.get(filter_mesh.remote(data_root, item, max_ratio))
        tasks.append(task)

    results = ray.get(tasks)
    valid_ids = sum(results,[])

    with open("valid_planar_shapes_except_cube.txt", "w") as f:
        for item in range(len(valid_ids)):
            if valid_ids[item]:
                f.write(ids[item] + "\n")

if __name__ == '__main__':
    test_obj(data_root, "planar_shapes_except_cube.txt")



