import math
import shutil

import open3d as o3d
import scipy
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import os
import numpy as np
from trimesh.visual.color import colors_to_materials

from shared.common_utils import safe_check_dir

Soft_Aesthetic = np.array([
    [182, 81, 73, 255],
    [224, 155, 128, 255],
    [232, 223, 206, 255],
    [203, 134, 85, 255],
    [126, 142, 101, 255],
])

Soft_Angel = np.array([
    [221, 245, 249, 255],
    [198, 206, 227, 255],
    [255, 250, 237, 255],
    [240, 200, 200, 255],
    [255, 246, 223, 255],
])

Soft_big = np.array([
    [144, 157, 147, 255],
    [234, 217, 199, 255],
    [233, 169, 159, 255],
    [220, 194, 161, 255],
    [229, 199, 197, 255],
    # [214,214,206,255],
    [163, 182, 178, 255],
    [201, 163, 162, 255],
    # [223,225,222,255],
    [236, 181, 174, 255],
    [241, 214, 203, 255],
    [175, 191, 178, 255],
    [230, 190, 178, 255],
    [144, 162, 166, 255],
])

DC_color = np.array([
    [198, 101, 89, 255],
    [137, 195, 13, 255],
    [45, 164, 203, 255],
    [208, 63, 148, 255],
    [193, 199, 41, 255],
    [113, 116, 181, 255],
    [211, 49, 68, 255],
    [255, 125, 39, 255],
])

DC_color2 = np.array([
    [255, 133, 116, 255],
    [255, 175, 54, 255],
    [176, 255, 254, 255],
    [130, 134, 208, 255],
    [255, 86, 91, 255],
    [102, 183, 161, 255],
    [174, 247, 16, 255],
    [60, 217, 255, 255],
    [198, 101, 89, 255],
])

DC_color3 = np.array([
    [181, 140, 127],
    [124, 204, 231],
    [116, 116, 211],
    [227, 208, 91],
    [52, 174, 163],
    [80, 134, 195],
    [157, 202, 93],

    [91, 190, 152],
    [97, 156, 228],
    [148, 162, 218],
    [245, 152, 106],
    [102, 181, 153],
    [157, 115, 181],
])

Bright_color = np.array([
    [233, 76, 61],
    [239, 161, 0],
    [223, 131, 194],
    [225,143,89],
    [144, 172, 59],
    [1, 199, 86],
    [149, 184, 216],
    # [0, 99, 177],
    [72, 130, 177],
    [189, 112, 164],
    [238, 127, 169],
    [101,100,164],
])

color_table = Bright_color


def to_blender_color(c):
    c = min(max(0, c), 255) / 255
    return c / 12.92 if c < 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)


def from_blender_color(c):
    color = max(0.0, c * 12.92) if c < 0.0031308 else 1.055 * math.pow(c, 1.0 / 2.4) - 0.055
    return hex(max(min(int(color * 255 + 0.5), 255), 0))


def init_gt_2_align(mesh, v_out_path, prefix, id, rebuild_face=False):
    # mesh.update_faces(mesh.nondegenerate_faces())
    # mesh.remove_infinite_values()
    # mesh.remove_unreferenced_vertices()

    rotation = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
    rotation = np.concatenate([rotation, np.zeros([3, 1])], axis=1)
    rotation = np.concatenate([rotation, np.zeros([1, 4])], axis=0)
    # TO DC
    # mesh.apply_transform(rotation)

    global num_of_gt_component, unique_gt_index, gt_index, sampled_points_for_gt
    global viz_mesh_for_gt

    index = None
    index = np.asarray([item[1] for item in mesh.metadata["_ply_raw"]["face"]["data"]], dtype=np.int32)
    num_of_gt_component = np.unique(index).shape[0]
    unique_gt_index = np.unique(index)

    sampled_points_for_gt = [None for _ in range(num_of_gt_component)]
    viz_mesh_for_gt = [None for _ in range(num_of_gt_component)]
    for i in unique_gt_index:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces[index == i])
        viz_mesh_for_gt[i] = o3d_mesh
        sampled_mesh = o3d_mesh.sample_points_uniformly(number_of_points=math.ceil(o3d_mesh.get_surface_area() * 1000))
        sampled_points_for_gt[i] = sampled_mesh

    index = index % color_table.shape[0]
    max_index = index.max()

    if rebuild_face:
        triangles = mesh.vertices[mesh.faces]
        mesh.vertices = triangles.reshape(-1, 3)
        mesh.faces = np.arange(len(mesh.vertices)).reshape(-1, 3)

    with open(os.path.join(v_out_path, "{}_{}.obj".format(id, prefix)), "w") as f:
        f.write("mtllib mat.mtl\n")
        for v in mesh.vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for idx in range(max_index + 1):
            f.write("usemtl mat_{}\n".format(idx))
            for face in mesh.faces[index == idx]:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


def align_color_with_gt(mesh, v_out_path, prefix, id, rebuild_face=False, component_type="color"):
    # mesh.update_faces(mesh.nondegenerate_faces())
    # mesh.remove_infinite_values()
    # mesh.remove_unreferenced_vertices()

    rotation = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
    rotation = np.concatenate([rotation, np.zeros([3, 1])], axis=1)
    rotation = np.concatenate([rotation, np.zeros([1, 4])], axis=0)
    # TO DC
    # mesh.apply_transform(rotation)

    index = None
    if component_type == "component":
        index = trimesh.graph.connected_component_labels(mesh.face_adjacency)
    elif component_type == "color":
        colors = np.asarray(mesh.visual.face_colors)
        index = np.unique(colors, axis=0, return_inverse=True)[1]
    elif component_type == "primitive":
        index = np.asarray([item[4] for item in mesh.metadata["_ply_raw"]["face"]["data"]], dtype=np.int32)

    unique_index = np.unique(index)
    num_of_component_2_align = unique_index.shape[0]

    sampled_points_for_2_align = [None for _ in range(num_of_component_2_align)]
    viz_mesh_for_2_align = [None for _ in range(num_of_component_2_align)]
    for i, idx in enumerate(unique_index):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces[index == idx])
        viz_mesh_for_2_align[i] = o3d_mesh
        if o3d_mesh.get_surface_area() == 0.:
            sampled_mesh = o3d.geometry.PointCloud()
            sampled_mesh.points = o3d.utility.Vector3dVector(mesh.vertices[mesh.faces[index == idx][0]])
        else:
            sampled_mesh = o3d_mesh.sample_points_uniformly(
                number_of_points=max(1, math.ceil(o3d_mesh.get_surface_area() * 1000)))
        sampled_points_for_2_align[i] = sampled_mesh

    distance_matrix = np.zeros([num_of_gt_component, num_of_component_2_align])

    for i in unique_gt_index:
        for j in range(unique_index.shape[0]):
            dist1 = np.asarray(sampled_points_for_gt[i].compute_point_cloud_distance(sampled_points_for_2_align[j]))
            dist2 = np.asarray(sampled_points_for_2_align[j].compute_point_cloud_distance(sampled_points_for_gt[i]))
            distance_matrix[i, j] = (np.mean(dist1) + np.mean(dist2)) / 2
            pass

    # Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(distance_matrix)

    new_index = np.zeros_like(index)
    for i in range(len(row_ind)):
        # assign row_ind[i] in gt's color to col_ind[i] in mesh's color
        # o3d.io.write_triangle_mesh("temp_gt.ply", viz_mesh_for_gt[row_ind[i]])
        # o3d.io.write_triangle_mesh("temp_pred.ply", viz_mesh_for_2_align[col_ind[i]])
        new_index[index == unique_index[col_ind[i]]] = row_ind[i]
        pass

    index = new_index
    index = index % color_table.shape[0]
    max_index = index.max()

    if rebuild_face:
        triangles = mesh.vertices[mesh.faces]
        mesh.vertices = triangles.reshape(-1, 3)
        mesh.faces = np.arange(len(mesh.vertices)).reshape(-1, 3)

    with open(os.path.join(v_out_path, "{}_{}.obj".format(id, prefix)), "w") as f:
        f.write("mtllib mat.mtl\n")
        for v in mesh.vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for idx in range(max_index + 1):
            f.write("usemtl mat_{}\n".format(idx))
            for face in mesh.faces[index == idx]:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


def save_mesh_tex(mesh, v_out_path, prefix, id, rebuild_face=False, color_by_component=False):
    mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    if color_by_component:
        id_face = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        colors = DC_color[id_face % DC_color.shape[0]]
    else:
        colors = np.asarray(mesh.visual.face_colors)
        index = np.unique(colors, axis=0, return_inverse=True)[1]
        colors = DC_color[index % DC_color.shape[0]]

    if rebuild_face:
        triangles = mesh.vertices[mesh.faces]
        mesh.vertices = triangles.reshape(-1, 3)
        mesh.faces = np.arange(len(mesh.vertices)).reshape(-1, 3)
    mesh.visual.face_colors = colors
    mesh.visual.vertex_colors = trimesh.visual.color.face_to_vertex_color(mesh, colors)
    mat = mesh.visual.to_texture()
    mat.material.ambient = np.array([255, 255, 255, 255], dtype=np.uint8)
    mat.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
    mat.material.specular = np.array([0, 0, 0, 0], dtype=np.uint8)
    mat.material.name = "{}_{}_mat".format(id, prefix)
    mesh.visual = mat

    mesh.export(os.path.join(v_out_path, "{}_{}.obj".format(id, prefix)),
                include_texture=True, write_texture=True,
                mtl_name="{}_{}_mat.mtl".format(id, prefix)
                )


def save_mesh(mesh, v_out_path, prefix, id, rebuild_face=False, color_by_component=False):
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    rotation = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
    rotation = np.concatenate([rotation, np.zeros([3, 1])], axis=1)
    rotation = np.concatenate([rotation, np.zeros([1, 4])], axis=0)
    # TO DC
    # mesh.apply_transform(rotation)

    index = None
    if color_by_component:
        index = trimesh.graph.connected_component_labels(mesh.face_adjacency)
    else:
        colors = np.asarray(mesh.visual.face_colors)
        index = np.unique(colors, axis=0, return_inverse=True)[1]

    index = index % color_table.shape[0]
    max_index = index.max()

    if rebuild_face:
        triangles = mesh.vertices[mesh.faces]
        mesh.vertices = triangles.reshape(-1, 3)
        mesh.faces = np.arange(len(mesh.vertices)).reshape(-1, 3)

    with open(os.path.join(v_out_path, "{}_{}.obj".format(id, prefix)), "w") as f:
        f.write("mtllib mat.mtl\n")
        for v in mesh.vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for idx in range(max_index + 1):
            f.write("usemtl mat_{}\n".format(idx))
            for face in mesh.faces[index == idx]:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


if __name__ == '__main__':
    if True:
        mesh = trimesh.load(r"G:/Dataset/GSP/Results/App/0044/Sewed_shape.ply")

        if True:
            index = np.asarray([item[1] for item in mesh.metadata["_ply_raw"]["face"]["data"]], dtype=np.int32)
        elif False:
            index = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        elif True:
            colors = np.asarray(mesh.visual.face_colors)
            index = np.unique(colors, axis=0, return_inverse=True)[1]

        index = index % color_table.shape[0]
        max_index = index.max()

        with open("G:/Dataset/GSP/Results/App/0044/Sewed_shape.obj", "w") as f:
            f.write("mtllib mat.mtl\n")
            for v in mesh.vertices:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for idx in range(max_index + 1):
                f.write("usemtl mat_{}\n".format(idx))
                for face in mesh.faces[index == idx]:
                    f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

        with open("G:/Dataset/GSP/Results/App/0044/mat.mtl", "w") as f:
            for idx in range(color_table.shape[0]):
                f.write("newmtl mat_{}\n".format(idx))
                f.write(
                    "Ka {} {} {}\n".format(color_table[idx, 0] / 255, color_table[idx, 1] / 255,
                                           color_table[idx, 2] / 255))
                f.write(
                    "Kd {} {} {}\n".format(color_table[idx, 0] / 255, color_table[idx, 1] / 255,
                                           color_table[idx, 2] / 255))
                f.write("Ks {} {} {}\n".format(0, 0, 0))
                f.write("Ns {}\n".format(1))

    random_flag = -1
    output_root = r"G:/Dataset/GSP/Results/viz_output/0120_random_mesh"
    viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis_random.txt").readlines()]

    # output_root = r"G:/Dataset/GSP/Results/viz_output/test2/Bright_color"
    # viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis6.txt").readlines()]

    # output_root = r"G:/Dataset/GSP/Results/viz_output/test2/Bright_color_random"
    # viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis_random6.txt").readlines()]

    safe_check_dir(output_root)
    gt_root = r"G:/Dataset/GSP/test_data_final/mesh"
    ours_root = r"G:/Dataset/GSP/Results/Ours/0112_total_mesh"
    hp_root = r"G:/Dataset/GSP/Results/Baselines/HP_mesh_1228"
    sed_root = r"G:/Dataset/GSP/Results/Baselines/SED_mesh_1228"
    complex_root = r"G:/Dataset/GSP/Results/Baselines/ComplexGen_prediction_0102"
    # viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/List/test_ids_small.txt").readlines()]
    # viz_id = [file.strip() for file in open(r"G:/Dataset/GSP/List/vis31.txt").readlines()]
    viz_id = sorted(viz_id)

    if random_flag > 0:
        num_step = len(viz_id) // random_flag
        viz_id = viz_id[::num_step][:random_flag]
        with open("G:/Dataset/GSP/List/viz_ids_random.txt", "w") as f:
            for item in viz_id:
                f.write(item + "\n")

    with open(os.path.join(output_root, "mat.mtl"), "w") as f:
        for idx in range(color_table.shape[0]):
            f.write("newmtl mat_{}\n".format(idx))
            # f.write(
            #     "Ka {} {} {}\n".format(
            #         color_table[idx, 0] / 255,
            #         color_table[idx, 1] / 255,
            #         color_table[idx, 2] / 255)
            # )
            # f.write(
            #     "Kd {} {} {}\n".format(
            #         color_table[idx, 0] / 255,
            #         color_table[idx, 1] / 255,
            #         color_table[idx, 2] / 255)
            # )
            f.write(
                "Ka {} {} {}\n".format(
                    to_blender_color(color_table[idx, 0]),
                    to_blender_color(color_table[idx, 1]),
                    to_blender_color(color_table[idx, 2]))
            )
            f.write(
                "Kd {} {} {}\n".format(
                    to_blender_color(color_table[idx, 0]),
                    to_blender_color(color_table[idx, 1]),
                    to_blender_color(color_table[idx, 2]))
            )

            f.write("Ks {} {} {}\n".format(0, 0, 0))
            f.write("Ns {}\n".format(1))

    # viz_id = ["00990738"]

    for file in tqdm(viz_id):
        prev_id = file

        gt_mesh = trimesh.load(os.path.join(gt_root, prev_id + ".ply"))
        init_gt_2_align(gt_mesh, output_root, "gt", prev_id, rebuild_face=True)

        shutil.copyfile(os.path.join(gt_root, "../gt/viz_curve_and_vertex/{}_corner.obj".format(prev_id)),
                        os.path.join(output_root, "{}_gt_corner.obj".format(prev_id)))
        shutil.copyfile(os.path.join(gt_root, "../gt/viz_curve_and_vertex/{}_curve.obj".format(prev_id)),
                        os.path.join(output_root, "{}_gt_curve.obj".format(prev_id)))
        # continue
        ours_mesh = trimesh.load(os.path.join(ours_root, prev_id, "mesh/0total_mesh.ply"))
        align_color_with_gt(ours_mesh, output_root, "ours", prev_id,
                            rebuild_face=False, component_type="primitive")
        shutil.copyfile(os.path.join(ours_root, prev_id, "viz_curve_and_vertex/viz_corners.obj"),
                        os.path.join(output_root, "{}_ours_corner.obj".format(prev_id)))
        shutil.copyfile(os.path.join(ours_root, prev_id, "viz_curve_and_vertex/viz_curves.obj"),
                        os.path.join(output_root, "{}_ours_curve.obj".format(prev_id)))
        # continue
        hp_mesh = trimesh.load(os.path.join(hp_root, prev_id, "clipped/mesh_transformed.ply"))
        align_color_with_gt(hp_mesh, output_root, "hp", prev_id,
                            rebuild_face=False, component_type="color")
        shutil.copyfile(os.path.join(hp_root, prev_id, "clipped/viz_corners.obj"),
                        os.path.join(output_root, "{}_hp_corner.obj".format(prev_id)))
        shutil.copyfile(os.path.join(hp_root, prev_id, "clipped/viz_curves.obj"),
                        os.path.join(output_root, "{}_hp_curve.obj".format(prev_id)))

        sed_mesh = trimesh.load(os.path.join(sed_root, prev_id, "clipped/mesh_transformed.ply"))
        align_color_with_gt(sed_mesh, output_root, "sed", prev_id,
                            rebuild_face=False, component_type="color")
        shutil.copyfile(os.path.join(sed_root, prev_id, "clipped/viz_corners.obj"),
                        os.path.join(output_root, "{}_sed_corner.obj".format(prev_id)))
        shutil.copyfile(os.path.join(sed_root, prev_id, "clipped/viz_curves.obj"),
                        os.path.join(output_root, "{}_sed_curve.obj".format(prev_id)))

        complex_mesh = trimesh.load(os.path.join(complex_root, "cut_grouped/" + prev_id + "_cut_grouped.ply"))
        align_color_with_gt(complex_mesh, output_root, "complex", prev_id,
                            rebuild_face=False, component_type="component")
        shutil.copyfile(os.path.join(complex_root, "viz_corner_curve/{}/vertices.obj".format(prev_id)),
                        os.path.join(output_root, "{}_complex_corner.obj".format(prev_id)))
        shutil.copyfile(os.path.join(complex_root, "viz_corner_curve/{}/curves.obj".format(prev_id)),
                        os.path.join(output_root, "{}_complex_curve.obj".format(prev_id)))

        # break
