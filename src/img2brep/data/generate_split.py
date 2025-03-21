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
                valid_ids[idx] = False
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


@ray.remote(num_cpus=1)
def get_split_(folders):
    valid_ids = []
    cube_ids = []
    for folder in folders:
        for ff in os.listdir(data_root / folder):
            if ff.endswith(".yml") and "features" in ff:
                is_pure_plane = True
                str = open(os.path.join(data_root, folder, ff), "r").read()
                pos = -1
                num_plane = 0
                num_line = 0
                while True:
                    pos = str.find("type:", pos + 1)
                    if pos == -1:
                        break

                    # if str[pos + 6:pos + 8] not in ["Li", "Pl"]:
                    #     is_pure_plane = False
                    #     break
                    if str[pos + 6:pos + 8] == "Pl":
                        num_plane += 1
                    else:
                        num_line += 1

                if not is_pure_plane:
                    continue

                if num_plane == 6 and num_line == 12:
                    cube_ids.append(folder)
                valid_ids.append(folder)

    return valid_ids, cube_ids


# Test all files under data_root
# - Filter out empty meshes
# - Filter out non-planar meshes
# - Filter out cube meshes
def get_splits(data_root):
    ray.init()

    folders = os.listdir(data_root)
    folders.sort()
    folders = folders[0:10000]

    num_batches = 80  # Larger than number of cpus for a more efficient task assignment
    batch_size = len(folders) // num_batches + 1

    valid_ids = []
    cube_ids = []

    tasks = []
    for i in range(num_batches):
        tasks.append(get_split_.remote(folders[i * batch_size:min((i + 1) * batch_size, len(folders))]))

    results = ray.get(tasks)
    for item in results:
        valid_ids += item[0]
        cube_ids += item[1]
    valid_ids = sorted(list(set(valid_ids)))
    cube_ids = sorted(list(set(cube_ids)))

    with open("../planar_shapes_id.txt", "w") as f:
        for item in valid_ids:
            f.write(item + "\n")

    with open("../cube_ids.txt", "w") as f:
        for item in cube_ids:
            f.write(item + "\n")

    with open("../planar_shapes_except_cube.txt", "w") as f:
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

    num_batches = 80  # Larger than number of cpus for a more efficient task assignment
    batch_size = len(ids) // num_batches + 1

    tasks = []
    for idx in range(num_batches):
        task = filter_mesh.remote(data_root, ids[batch_size * idx:min(len(ids), batch_size * (idx + 1))], max_ratio)
        # result = ray.get(filter_mesh.remote(data_root, item, max_ratio))
        tasks.append(task)

    results = ray.get(tasks)
    valid_ids = sum(results, [])

    with open("../valid_planar_shapes_except_cube.txt", "w") as f:
        for item in range(len(valid_ids)):
            if valid_ids[item]:
                f.write(ids[item] + "\n")

@ray.remote(num_cpus=1)
def remove_multiple_component_ids_(v_root, folders):
    valid_flags = [False] * len(folders)
    for idx, v_folder in enumerate(folders):
        for ff in os.listdir(v_root / v_folder):
            if not ff.endswith(".obj"):
                continue
            mesh = trimesh.load_mesh(v_root / v_folder / ff, process=False, maintain_order=True)
            if mesh.faces.shape[0] < 10 or mesh.vertices.shape[0] < 10:
                continue

            if mesh.split().shape[0] != 1:
                continue

            valid_flags[idx]=True

    return valid_flags

# Test all files under data_root
# - Record shape id that without an obj file
# - Record shape id that contains multiple parts
def remove_multiple_component_ids(v_id_cubes=None):
    ray.init(
        # local_mode=True,
        # num_cpus=1,
    )

    folders = os.listdir(data_root)
    folders.sort()
    folders = folders[0:100]

    # Except for the cubes
    if v_id_cubes is not None:
        cube_ids = [item.strip() for item in open(v_id_cubes).readlines()]
        folders = list(set(folders) - set(cube_ids))
        folders.sort()
        print("Has {} ids after removing {} cubes".format(len(folders), len(cube_ids)))

    num_batches = 80  # Larger than number of cpus for a more efficient task assignment
    batch_size = len(folders) // num_batches + 1

    tasks = []
    for idx in range(num_batches):
        task = remove_multiple_component_ids_.remote(
            data_root, folders[batch_size * idx:min(len(folders), batch_size * (idx + 1))])
        # result = ray.get(filter_mesh.remote(data_root, item, max_ratio))
        tasks.append(task)

    results = ray.get(tasks)
    valid_ids = sum(results, [])

    with open("id_shapes_with_multiple_component_or_few_faces.txt", "w") as f:
        for item in range(len(valid_ids)):
            if not valid_ids[item]:
                f.write(folders[item] + "\n")

@ray.remote(num_cpus=1)
def remove_other_shape_(v_root, v_folders):
    valid_ids = [True] * len(v_folders)
    for idx, folder in enumerate(v_folders):
        for ff in os.listdir(v_root / folder):
            if ff.endswith(".yml") and "features" in ff:
                str = open(os.path.join(data_root, folder, ff), "r").read()
                pos = str.find("type: Ot", 0)
                if pos != -1:
                    valid_ids[idx] = False
                    break

                pos = str.find("type: Ex", 0)
                if pos != -1:
                    valid_ids[idx] = False
                    break

                pos = str.find("type: Re", 0)
                if pos != -1:
                    valid_ids[idx] = False
                    break

    return valid_ids

# Test all files under data_root
# - Record shape id that has type "other"
def remove_other_shape():
    ray.init(
        # local_mode=True,
        # num_cpus=1,
        dashboard_port=15000,
        dashboard_host="0.0.0.0"
    )

    folders = os.listdir(data_root)
    folders.sort()
    folders = folders

    num_batches = 80  # Larger than number of cpus for a more efficient task assignment
    batch_size = len(folders) // num_batches + 1

    tasks = []
    for idx in range(num_batches):
        task = remove_other_shape_.remote(
            data_root, folders[batch_size * idx:min(len(folders), batch_size * (idx + 1))])
        # result = ray.get(filter_mesh.remote(data_root, item, max_ratio))
        tasks.append(task)

    results = ray.get(tasks)
    valid_ids = sum(results, [])

    with open("id_shapes_with_others.txt", "w") as f:
        for item in range(len(valid_ids)):
            if not valid_ids[item]:
                f.write(folders[item] + "\n")

if __name__ == '__main__':
    # remove_multiple_component_ids(r"C:/repo/python/src/img2brep/data/cube_ids_in_abc.txt")
    remove_other_shape()
    # get_splits(data_root)
    # test_obj(data_root, "planar_shapes_except_cube.txt")
