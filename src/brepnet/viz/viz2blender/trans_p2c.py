import os
import shutil

import numpy as np
from tqdm import tqdm
import trimesh

import ray

# data_root = r"E:\data\img2brep\1202\sed_point2cad"
# prefix = "sed"
data_root = r"E:\data\img2brep\1202\hp_point2cad"
prefix = "hp"
out_root = data_root + "_models"


def save_curves_as_obj(curves, output_path):
    """
    将多个曲线按两个点一组逐一连接并存储为 OBJ 文件。

    Args:
        curves (list of ndarray): 每个元素是一个 N x 3 的 NumPy 数组，表示一条曲线的采样点。
        output_path (str): 输出的 OBJ 文件路径。
    """
    with open(output_path, 'w') as obj_file:
        vertex_offset = 0  # 用于处理顶点索引偏移
        for curve in curves:
            # 写入顶点
            for point in curve:
                obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # 写入线段（逐对连接）
            for i in range(len(curve) - 1):
                if np.linalg.norm(curve[i] - curve[i + 1]) > 3e-3:
                    continue
                obj_file.write(f"l {vertex_offset + i + 1} {vertex_offset + i + 2}\n")

            # 更新顶点偏移量
            vertex_offset += len(curve)


def process_one(data_root, out_root, folder_name, prefix):
    folder_path = str(os.path.join(data_root, folder_name))
    folder_name = f"{prefix}_{folder_name}"
    local_save_root = str(os.path.join(out_root, folder_name))
    if os.path.exists(local_save_root):
        shutil.rmtree(local_save_root)
    os.makedirs(local_save_root, exist_ok=True)

    # model: use trimesh to read the ply mesh and save in obj format
    model_path = os.path.join(folder_path, "clipped", "mesh_transformed.ply")
    if os.path.exists(model_path):
        model = trimesh.load(model_path)
        model.export(os.path.join(local_save_root, f"{folder_name}_model.obj"))

    # vertex
    vertex_path = os.path.join(folder_path, "clipped", "remove_duplicates_corners.ply")
    if os.path.exists(vertex_path):
        vertex = trimesh.load(vertex_path)
        vertex.export(os.path.join(local_save_root, f"{folder_name}_vertex.obj"))

    # wire
    wire_path = os.path.join(folder_path, "clipped", "curve_points.xyzc")
    if os.path.exists(wire_path):
        xyzc_data = np.loadtxt(wire_path)
        all_c = set(xyzc_data[:, -1])
        xyz_list = []
        for c in all_c:
            xyz = xyzc_data[xyzc_data[:, -1] == c][:, :-1]
            xyz_list.append(xyz)
        save_curves_as_obj(xyz_list, os.path.join(local_save_root, f"{folder_name}_wire.obj"))
    pass


if __name__ == "__main__":
    all_folders = os.listdir(data_root)
    all_folders.sort()

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)
    # for folder in tqdm(all_folders):
    #     process_one(data_root, out_root, folder)

    ray.init(local_mode=False)
    process_one_remote = ray.remote(process_one)
    futures = []
    for folder in tqdm(all_folders):
        futures.append(process_one_remote.remote(data_root, out_root, folder, prefix))

    for f in tqdm(futures):
        ray.get(f)
