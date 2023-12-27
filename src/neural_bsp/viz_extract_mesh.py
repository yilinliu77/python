import open3d as o3d
import scipy
import trimesh
from tqdm import tqdm
import os
import numpy as np

from shared.common_utils import safe_check_dir

color_table = np.array([
    [246,189,96,255],
    [247,237,226,255],
    [245,202,195,255],
    [132,165,157,255],
    [242,132,130,255],
])

def save_mesh(v_path, v_out_path):
    mesh = trimesh.load(v_path)
    mesh.export(v_out_path, include_texture=True)

if __name__ == '__main__':
    output_root = r"G:/Dataset/GSP/viz_output"
    safe_check_dir(output_root)
    gt_root = r"G:/Dataset/GSP/test_data_whole3/mesh"
    ours_root = r"G:/Dataset/GSP/Baselines/1219_v15+_parsenet_unet_base16_focal75_wonormal_channel4_float32_mesh"
    hp_root = r"G:/Dataset/GSP/Baselines/HP_mesh_1214"
    sed_root = r"G:/Dataset/GSP/Baselines/SED_mesh_1218"
    complex_root = r"G:/Dataset/GSP/Baselines/ComplexGen_prediction_1222"
    viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/Baselines/viz_ids.txt").readlines()]
    viz_id = sorted(viz_id)

    for file in tqdm(viz_id):
        prev_id = file
        mesh = trimesh.load(os.path.join(gt_root, prev_id + ".ply"))
        colors = mesh.visual.face_colors
        triangles = mesh.vertices[mesh.faces]
        mesh.vertices = triangles.reshape(-1, 3)
        mesh.faces = np.arange(len(mesh.vertices)).reshape(-1, 3)
        mesh.visual.vertex_colors = colors
        mesh.export(os.path.join(output_root, prev_id + "_gt.obj"))

        mesh = trimesh.load(os.path.join(ours_root, prev_id, "mesh/0total_mesh.ply"))
        mesh.export(os.path.join(output_root, prev_id + "_ours.obj"))

        mesh = trimesh.load(os.path.join(hp_root, prev_id, "clipped/mesh_transformed.ply"))
        mesh.export(os.path.join(output_root, prev_id + "_hp.obj"))

        mesh = trimesh.load(os.path.join(sed_root, prev_id, "clipped/mesh_transformed.ply"))
        mesh.export(os.path.join(output_root, prev_id + "_sed.obj"))

        mesh = trimesh.load(os.path.join(complex_root, "cut_grouped/"+prev_id+"_cut_grouped.ply"))
        # Assign each component a color
        id_face = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        colors = color_table[id_face % color_table.shape[0]]
        mesh.visual.face_colors = colors
        mesh.export(os.path.join(output_root, prev_id + "_complex.obj"))

