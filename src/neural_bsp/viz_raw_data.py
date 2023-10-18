import shutil

import open3d as o3d
import os
import numpy as np
from tqdm import tqdm

from src.neural_bsp.abc_hdf5_dataset import normalize_points

def viz_and_select():
    v_root = r"G:\Dataset\ABC\quadric_test"
    # quadric_file = r"G:\Dataset\GSP\quadric_ids.txt"
    quadric_file = r"G:\Dataset\ABC\quadric_test_viz_ids.txt"
    all_quadrics = [item.strip() for item in open(quadric_file).readlines() if item.strip().startswith("0099")]

    accept_list = []

    for file in tqdm(all_quadrics):
        model_name = [item for item in os.listdir(os.path.join(v_root, file)) if item.endswith(".obj")]
        if len(model_name) == 0:
            continue
        print(model_name[0])
        mesh = o3d.io.read_triangle_mesh(os.path.join(v_root, file, model_name[0]))
        points = np.asarray(mesh.vertices)
        points = normalize_points(points)
        mesh.vertices = o3d.utility.Vector3dVector(points)

        # o3d.visualization.draw_geometries([mesh],
        #                                   mesh_show_wireframe=True
        #                                   )

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        def key_callback1(vis):
            accept_list.append(model_name[0])
            vis.destroy_window()

        def key_callback2(vis):
            vis.destroy_window()

        vis.register_key_callback(65, key_callback1)
        vis.register_key_callback(68, key_callback2)
        vis.add_geometry(mesh)
        vis.add_geometry(o3d.geometry.LineSet.create_from_triangle_mesh(mesh))
        vis.run()

def copy_test_files():
    input_dir = r"G:\Dataset\ABC\last_chunk"
    output_dir = r"G:\Dataset\ABC\quadric_test"

    tasks = [item.strip() for item in open(r"G:\Dataset\ABC\quadric_test_viz_ids.txt").readlines()]
    for task in tqdm(tasks):
        shutil.copytree(os.path.join(input_dir, task), os.path.join(output_dir, task))

if __name__ == '__main__':
    # viz_and_select()
    copy_test_files()