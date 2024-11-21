import os
from pathlib import Path
import random

from tqdm import tqdm
import argparse
import numpy as np
import shutil
import ray
import glob

from src.brepnet.eval.check_valid import check_step_valid_soild
from src.brepnet.post.utils import *
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_SOLID
import trimesh

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def random_rgba():
    return tuple(np.random.randint(100, 256, 3).tolist() + [255])


def idx_to_rgba(idx):
    r = idx // 100
    g = (idx // 10) % 10
    b = idx % 10
    return (r + 100, g + 100, b + 100, 255)


def rgb_to_idx(rgb):
    r, g, b = rgb
    return (r - 100) * 100 + (g - 100) * 10 + (b - 100)


def normalize_mesh(mesh):
    bounds = mesh.bounds
    scale = 2.0 / (bounds[1] - bounds[0]).max()
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)
    return mesh


def arrange_meshes(files, out_path, intervals=0.5, color_mode="index"):
    assert color_mode in ["random", "index"]
    if type(files[0]) == str:
        meshes = [trimesh.load(file) for file in files]
    elif type(files[0]) == trimesh.Trimesh:
        meshes = [mesh for mesh in files]
    else:
        raise ValueError("Invalid input type")
    num_meshes = len(meshes)

    grid_size = int(np.ceil(np.sqrt(num_meshes)))
    combined = []
    for idx, mesh in enumerate(meshes):
        row = idx // grid_size
        col = idx % grid_size
        translation = [col * (2 + intervals), -row * (2 + intervals), 0]
        if color_mode == "index":
            mesh.visual.face_colors = np.array([idx_to_rgba(idx)] * len(mesh.faces))
        elif color_mode == "random":
            mesh.visual.face_colors = np.array([random_rgba()] * len(mesh.faces))
        else:
            raise ValueError("Invalid color mode")
        mesh.apply_translation(translation)
        combined.append(mesh)

    combined_mesh = trimesh.util.concatenate(combined)
    combined_mesh.export(out_path)


def arrange_meshes_row(file_paths, out_path, intervals=0.5, color_mode="random"):
    assert color_mode in ["random", "index"]
    meshes = [normalize_mesh(trimesh.load(file)) for file in file_paths]
    num_meshes = len(meshes)

    combined = []
    for idx, mesh in enumerate(meshes):
        translation = [idx * (2 + intervals), 0, 0]
        if color_mode == "index":
            mesh.visual.face_colors = np.array([idx_to_rgba(idx)] * len(mesh.faces))
        elif color_mode == "random":
            mesh.visual.face_colors = np.array([random_rgba()] * len(mesh.faces))
        else:
            raise ValueError("Invalid color mode")
        mesh.apply_translation(translation)
        combined.append(mesh)

    combined_mesh = trimesh.util.concatenate(combined)
    combined_mesh.export(out_path)


def explore_primitive(shape, primitive):
    primitive_list = []
    explorer = TopExp_Explorer(shape, primitive)
    while explorer.More():
        primitive_list.append(explorer.Current())
        explorer.Next()

    return primitive_list


def compute_solid_complexity(file_path, num_samples=4):
    try:
        shape = read_step_file(file_path, as_compound=False, verbosity=False)
    except:
        return {"is_valid_solid": False, "mean_curvature": -1, "num_faces": -1, "num_edges": -1, "num_vertices": -1}

    is_valid = check_step_valid_soild(file_path)

    sample_point_curvature = []

    face_list = explore_primitive(shape, TopAbs_FACE)
    edge_list = explore_primitive(shape, TopAbs_EDGE)
    vetex_list = explore_primitive(shape, TopAbs_VERTEX)

    for face in face_list:
        surf_adaptor = BRepAdaptor_Surface(face)
        u_min, u_max, v_min, v_max = (surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter(), surf_adaptor.FirstVParameter(),
                                      surf_adaptor.LastVParameter())

        u_samples = np.linspace(u_min, u_max, int(np.sqrt(num_samples)))
        v_samples = np.linspace(v_min, v_max, int(np.sqrt(num_samples)))

        for u in u_samples:
            for v in v_samples:
                props = BRepLProp_SLProps(surf_adaptor, u, v, 2, 1e-6)
                if props.IsCurvatureDefined():
                    mean_curvature = props.MeanCurvature()
                    sample_point_curvature.append(abs(mean_curvature))

    if len(sample_point_curvature) > 0:
        mean_curvature = np.mean(sample_point_curvature)
    else:
        mean_curvature = 0.0

    if mean_curvature < 1e-4:
        mean_curvature = 0.0

    return {"is_valid_solid": is_valid, "mean_curvature": mean_curvature,
            "num_faces"     : len(face_list), "num_edges": len(edge_list), "num_vertices": len(vetex_list)}


compute_solid_complexity_remote = ray.remote(compute_solid_complexity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--sort", action='store_true')
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--index", action='store_true')
    args = parser.parse_args()
    data_root = args.data_root
    out_root = args.out_root
    random = args.random
    sort = args.sort
    sample_num = args.sample_num
    use_ray = args.use_ray
    onlyvalid = args.valid

    # can only choose one of random, sort, seg
    if sum([random, sort]) != 1:
        raise ValueError("Only and must set one of random, sort")

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root)
    folder_names = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    # folder_names = folder_names[:10]
    if random:
        sample_folder = random.sample(folder_names, sample_num)
        for folder in tqdm(sample_folder):
            shutil.copytree(str(os.path.join(data_root, folder)), str(os.path.join(out_root, folder)))
        exit(0)

    # accumulate the data(mean_curvature, num_faces, num_edges, num_vertices) of each sample
    folder_names.sort()
    folder_scores = {}
    if not use_ray:
        pbar = tqdm(folder_names)
        for folder in pbar:
            pbar.set_description(f"Processing {folder}")
            file_path = glob.glob(os.path.join(data_root, folder, "*.step"))
            if len(file_path) == 0:
                continue
            file_path = file_path[0]
            score = compute_solid_complexity(file_path)
            folder_scores[folder] = score
    else:
        ray.init(
                # local_mode=True,
        )
        futures = []
        futures_folder_names = []
        for folder in tqdm(folder_names):
            file_path = glob.glob(os.path.join(data_root, folder, "*.step"))
            if len(file_path) == 0:
                continue
            file_path = file_path[0]
            futures.append(compute_solid_complexity_remote.remote(file_path))
            futures_folder_names.append(folder)

        for idx in tqdm(range(len(futures))):
            folder_name = futures_folder_names[idx]
            result = ray.get(futures[idx])
            folder_scores[folder_name] = result

    # viz the valid_ratio-face_num distribution (for all samples, not only valid)
    result_dict = {}
    for face_num in range(3, 31):
        result_dict[face_num] = []
    for folder, score in folder_scores.items():
        face_num = score["num_faces"]
        is_valid = score["is_valid_solid"]
        if face_num not in result_dict:
            continue
        result_dict[face_num].append(is_valid)

    plt.figure(figsize=(12, 6))
    face_num = [str(i).zfill(2) for i in range(3, 31)]
    sample_num = [len(result_dict[i]) for i in range(3, 31)]
    sns.barplot(x=face_num, y=sample_num)
    plt.title('Sample num by Faces num')
    plt.xlabel('Faces num')
    plt.xticks(rotation=45)
    plt.ylabel('Sample num')
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "sample_num_by_faces_num.png"))

    plt.figure(figsize=(12, 6))
    face_num = [str(i).zfill(2) for i in range(3, 31)]
    valid_ratio = []
    for i in range(3, 31):
        if len(result_dict[i]) == 0:
            face_num.remove(str(i).zfill(2))
        else:
            valid_ratio.append(np.sum(result_dict[i]) / len(result_dict[i]))
    sns.barplot(x=face_num, y=valid_ratio)
    plt.title('Valid ratio by Faces num')
    plt.xlabel('Faces num')
    plt.xticks(rotation=45)
    plt.ylabel('Valid ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "valid_ratio_by_faces_num.png"))

    if onlyvalid:
        valid_folder_scores = {}
        for folder, score in folder_scores.items():
            if score['is_valid_solid']:
                valid_folder_scores[folder] = score
        folder_scores = valid_folder_scores

    if sort:
        # sort the folders based on the mean_curvature, num_faces, num_edges, num_vertices
        sorted_folders = sorted(folder_scores.items(),
                                key=lambda x: (x[1]["num_faces"], x[1]["mean_curvature"], x[1]["num_edges"], x[1]["num_vertices"]),
                                reverse=True)

        for idx, folder in enumerate(tqdm(sorted_folders)):
            shutil.copytree(str(os.path.join(data_root, folder[0])), str(os.path.join(out_root, f"{idx:05d}_{folder[0]}")))
        ray.shutdown()
        mesh_path_list = glob.glob(os.path.join(out_root, "**", "*.stl"), recursive=True)
        mesh_path_list.sort()
        prefix = Path(data_root).name
        name = os.path.join(out_root, "{}.ply".format(prefix))
        if args.index:
            arrange_meshes(mesh_path_list, name, color_mode="index")
        else:
            arrange_meshes(mesh_path_list, name, color_mode="random")
        print(f"arranged mesh is saved to {name}")

    print("Done")
