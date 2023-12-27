import open3d as o3d
import scipy
import trimesh
from tqdm import tqdm
import os
import numpy as np

from shared.common_utils import safe_check_dir

def save_mesh(v_path, v_out_path):
    mesh = trimesh.load(v_path)
    mesh.export(v_out_path, include_texture=True)

if __name__ == '__main__':
    output_root = r"G:/Dataset/GSP/viz_output"
    safe_check_dir(output_root)
    gt_root = r"G:/Dataset/GSP/test_data_whole2/mesh"
    viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/Baselines/viz_ids.txt").readlines()]
    viz_id = sorted(viz_id)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1600,height=1600)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(54)
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    init_pos = np.array([0,0,2])
    rotation = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 0, 0], degrees=True)
    init_matrix = np.eye(4)
    init_matrix[:3,:3] = rotation.as_matrix()
    init_matrix[:3,3] = init_pos
    # init_matrix = np.linalg.inv(init_matrix)
    camera_params.extrinsic = init_matrix

    id_cur = 0
    def key_callback(vis):
        global id_cur, camera_params
        if id_cur!=0:
            ctr = vis.get_view_control()
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            matrix = np.asarray(camera_params.extrinsic).copy()
            matrix = np.linalg.inv(matrix)
            # matrix[:3,3] *= 0
            prev_id = viz_id[id_cur - 1]
            np.savetxt(os.path.join(output_root, prev_id + "_cameras.txt"), matrix)

        if id_cur >= len(viz_id):
            return

        file = viz_id[id_cur]
        prefix = file[:8]
        mesh_cur = o3d.io.read_triangle_mesh(os.path.join(gt_root, file + ".ply"))
        vis.clear_geometries()
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_cur)
        vis.add_geometry(mesh_cur)
        vis.add_geometry(lineset)
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = init_matrix
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        id_cur += 1
        return

    vis.register_key_callback(65, key_callback)
    vis.run()
