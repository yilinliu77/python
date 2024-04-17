import os
from functools import partial
from pathlib import Path

import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm

from shared.common_utils import normalize_vector
from thirdparty import cuda_distance
from multiprocessing import Pool

# raw_data_dir = r"G:\Dataset\ABC\last_chunk"
output_root = r"G:\Dataset\GSP\test_data2"
opj = os.path.join

def check_connected(v_prefix):
    mesh_file = [item for item in os.listdir(os.path.join(raw_data_dir, v_prefix)) if item.endswith(".obj")][0]

    mesh = trimesh.load_mesh(os.path.join(raw_data_dir, v_prefix, mesh_file), process=False, maintain_order=True)
    return mesh.split().shape[0] == 1

def generate_test_ids():
    ids = [item.strip() for item in open(r"G:\Dataset\GSP\quadric_ids.txt").readlines()]
    files = [item for item in os.listdir(raw_data_dir)]
    valid_tasks = list(set(files).intersection(set(ids)))
    valid_tasks = sorted(valid_tasks)

    check_connected(valid_tasks[0])

    pool = Pool(24)
    flags = pool.map(check_connected, valid_tasks)
    valid_tasks = [item for idx,item in enumerate(valid_tasks) if flags[idx]]
    with open(r"G:\Dataset\GSP\test_ids.txt", "w") as f:
        for item in valid_tasks:
            f.write(item + "\n")
    pass

################################################################################################

def normalize_points(v_points):
    min_xyz = v_points.min(axis=0)
    max_xyz = v_points.max(axis=0)
    diag = np.linalg.norm(max_xyz - min_xyz)
    center_xyz = (min_xyz + max_xyz) / 2
    v_points = (v_points - center_xyz[None, :]) / diag * 2
    return v_points

def generate_coords(v_resolution):
    coords = np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution), indexing="ij")
    coords = np.stack(coords, axis=3) / (v_resolution - 1)
    coords = (coords * 2 - 1).astype(np.float32)
    return coords

def read_mesh(v_file):
    trimesh_mesh = trimesh.load(v_file, process=False, maintain_order=True)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(trimesh_mesh.vertices),
        o3d.utility.Vector3iVector(trimesh_mesh.faces)
    )

    return mesh

def generate_udf(v_file, v_coords):
    mesh = read_mesh(v_file)
    mesh.compute_triangle_normals()

    points = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)
    triangles = points[faces]
    num_queries = v_coords.shape[0]
    query_result = cuda_distance.query(triangles.reshape(-1), v_coords.reshape(-1), 512, 512 ** 3)

    udf = np.asarray(query_result[0]).astype(np.float32)
    closest_points = np.asarray(query_result[1]).reshape((num_queries, 3)).astype(np.float32)
    dir = closest_points - v_coords
    length = np.linalg.norm(dir, axis=1)
    dir = normalize_vector(dir)

    normals = normals[query_result[2]].astype(np.float32)
    feat_data = np.concatenate([udf[..., None], dir, normals], axis=1)
    return feat_data

def generate_test_data_single(v_file, coords, v_output_root):
    prefix = Path(v_file).stem

    pc_ndc_file = opj(v_output_root, "ndc", "{}_pc.ply".format(prefix))
    pc_mesh_file = opj(output_root, "ndc", "{}_mesh.ply".format(prefix))
    pc_mesh_stash = opj(output_root, "ndc", "{}_mesh.obj".format(prefix))

    norm_mesh_file = opj(output_root, "mesh", "{}.ply".format(prefix))

    mesh_udf_file = opj(output_root, "mesh_udf", "{}".format(prefix))
    pc_udf_file = opj(output_root, "pc_udf", "{}".format(prefix))

    # Read model
    pc = o3d.io.read_point_cloud(opj(v_output_root, "poisson", v_file+".ply"))

    # NDC for clean pc
    pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points) / 2)
    o3d.io.write_point_cloud(pc_ndc_file, pc)
    os.chdir("E:/CODE/NDC")
    os.system(
        "python main.py --test_input {} --input_type pointcloud --method undc --postprocessing --point_num 10000 --grid_size 128 --output_name {}".format(
            pc_ndc_file,
            pc_mesh_stash))
    ndc_mesh = read_mesh(pc_mesh_stash)
    ndc_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(ndc_mesh.vertices) / 128 * 2 - 1
    )
    o3d.io.write_triangle_mesh(pc_mesh_file, ndc_mesh)

    feat_data = generate_udf(pc_mesh_file, coords)
    np.save(pc_udf_file, feat_data)

    # Generate GT UDF
    feat_data = generate_udf(norm_mesh_file, coords)
    np.save(mesh_udf_file, feat_data)

    return

def generate_test_data():
    coods = generate_coords(256).reshape(-1,3)

    files = [item.split(".")[0] for item in  os.listdir(opj(output_root, "mesh"))]

    os.makedirs(opj(output_root, "ndc"), exist_ok=True)
    os.makedirs(opj(output_root, "mesh_udf"), exist_ok=True)
    os.makedirs(opj(output_root, "pc_udf"), exist_ok=True)

    with Pool(1) as p:
        print(list(tqdm(p.imap(
            partial(generate_test_data_single, coords=coods, v_output_root=output_root),
            files), total=len(files))))
    # for item in tqdm(mesh_files):
    #     generate_test_data_single(item, coods, v_output_root)

    return

if __name__ == '__main__':
    generate_test_data()