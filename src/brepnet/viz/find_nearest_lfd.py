import glob

import ray, os
from lfd import LightFieldDistance
import trimesh
from tqdm import tqdm

from src.brepnet.eval.check_valid import load_data_with_prefix, check_step_valid_soild
from src.brepnet.viz.sort_and_merge import arrange_meshes
import argparse


def load_mesh(mesh_path):
    return trimesh.load(mesh_path)


load_mesh_remote = ray.remote(load_mesh)


def batch_load_mesh(mesh_paths):
    futures = [load_mesh_remote.remote(mesh_path) for mesh_path in tqdm(mesh_paths)]
    results = []
    for future in tqdm(futures):
        results.append(ray.get(future))
    return results
    # return ray.get([load_mesh_remote.remote(mesh_path) for mesh_path in tqdm(mesh_paths)])


# rest of code
def compute_lfd(mesh_ref: trimesh.Trimesh, mesh: trimesh.Trimesh) -> float:
    os.system('export DISPLAY=:0')
    lfd_calc = LightFieldDistance(verbose=False)
    return lfd_calc.get_distance(
            mesh_ref.vertices, mesh_ref.faces,
            mesh.vertices, mesh.faces
    )


compute_lfd_remote = ray.remote(compute_lfd)


def find_nearest_lfd(mesh_ref: trimesh.Trimesh, meshes: list[trimesh.Trimesh], local_out_root, meshes_prefix, topk=10, use_ray=True):
    if not use_ray:
        distances: list[float] = []
        for m in tqdm(meshes):
            distances.append(compute_lfd(mesh_ref, m))
    else:
        distances = []
        futures = [compute_lfd_remote.remote(mesh_ref, m) for m in tqdm(meshes)]
        for future in tqdm(futures):
            distance = ray.get(future)
            distances.append(distance)

    sorted_idx = sorted(range(len(distances)), key=lambda k: distances[k])
    topk_near_idx = sorted_idx[:topk]
    topk_near_meshes = [meshes[i] for i in topk_near_idx]
    # nearest_mesh: trimesh.Trimesh = meshes[distances.index(min(distances))]
    arrange_meshes([mesh_ref] + topk_near_meshes, os.path.join(local_out_root, "lfd_nearest.ply"))
    with open(os.path.join(local_out_root, "lfd_nearest.txt"), "w") as f:
        f.write("Source: {}\n".format(os.path.basename(local_out_root)))
        f.write("Top10 Nearest:\n")
        for idx in topk_near_idx:
            f.write(f"{meshes_prefix[idx]} {distances[idx]}\n")


find_nearest_lfd_remote = ray.remote(find_nearest_lfd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_post", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--txt", type=str, required=False, default="")

    args = parser.parse_args()
    fake_post = args.fake_post
    train_root = args.train_root
    ray.init()

    if not os.path.exists(fake_post) or not os.path.exists(train_root):
        print("fake_post: ", fake_post)
        print("train_root: ", train_root)
        raise ValueError("Invalid path")

    print("\nLoading fake mesh...")
    if args.prefix:
        all_folders = [os.path.join(fake_post, args.prefix)]
    else:
        all_folders = [f for f in os.listdir(fake_post) if os.path.isdir(os.path.join(fake_post, f))]
        all_folders.sort()
        all_folders = all_folders[:100]
    fake_mesh_paths = []
    for folder in tqdm(all_folders):
        local_root = os.path.join(fake_post, folder)
        step_file = glob.glob(os.path.join(local_root, "*.step"))
        if len(step_file) == 0:
            continue
        if not check_step_valid_soild(step_file[0]):
            continue
        stl_file = glob.glob(os.path.join(local_root, "*.stl"))
        if len(stl_file) == 0:
            continue
        fake_mesh_paths.append(stl_file[0])
    fake_meshes = batch_load_mesh(fake_mesh_paths)
    print(f"Loding {len(fake_meshes)} fake meshes")
    assert len(fake_meshes) > 0

    print("\nLoading training meshes...")
    if args.txt:
        with open(args.txt, "r") as f:
            all_trainin_folders = [line.strip() for line in f.readlines()]
    else:
        all_trainin_folders = [f for f in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, f))]
    training_meshes_paths = [os.path.join(train_root, folder, "mesh.ply") for folder in all_trainin_folders]
    training_meshes_paths = training_meshes_paths[:1000]
    training_meshes = batch_load_mesh(training_meshes_paths)
    print(f"Loding {len(training_meshes)} training meshes")
    assert len(training_meshes) > 0
    ray.shutdown()

    print("\nComputing LFD...")
    for idx, fake_mesh in enumerate(tqdm(fake_meshes)):
        local_root = os.path.dirname(fake_mesh_paths[idx])
        find_nearest_lfd(fake_mesh, training_meshes, local_root, all_trainin_folders, topk=10, use_ray=False)
        # find_nearest_lfd_remote.remote(fake_mesh, training_meshes, local_root, all_trainin_folders, topk=10, use_ray=True)
