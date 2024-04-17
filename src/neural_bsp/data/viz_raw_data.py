import math
import shutil

import open3d as o3d
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pyrender
from PIL import Image

from src.neural_bsp.abc_hdf5_dataset import normalize_points

def viz(v_folder):
    for file in tqdm(os.listdir(v_folder)):
        # print(file)
        pcd = o3d.io.read_point_cloud(os.path.join(v_folder, file))
        points = np.asarray(pcd.points)
        box_min = np.min(points, axis=0)
        box_max = np.max(points, axis=0)
        points = (points - (box_min+box_max)/2)/np.sqrt(((box_max - box_min)**2).sum())
        points = points * 2

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(points, colors=np.ones_like(points)))

        camera = pyrender.PerspectiveCamera(yfov=np.pi/2)
        rotation = Rotation.from_euler('xyz', [-45, 45, 0], degrees=True).as_matrix()
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = np.array([1., 1., 1.])
        scene.add(camera, pose=camera_pose)

        # Set up the off-screen renderer
        renderer = pyrender.OffscreenRenderer(512, 512)
        color, _ = renderer.render(scene)
        image = Image.fromarray(color)
        image.save(os.path.join(r"G:\Dataset\Complex\imgs","{}.jpg".format(file[:-4])))
        pass
        # o3d.visualization.draw_geometries([mesh],)


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
    # copy_test_files()
    viz(r"G:\Dataset\Complex\data\default\test_point_clouds")