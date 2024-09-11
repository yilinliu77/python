import os.path
import traceback

import numpy as np
import open3d as o3d
import tqdm
from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_Status
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from OCC.Core.TopoDS import TopoDS_Iterator

# from pytorch3d.loss import chamfer_distance

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud

from src.brepnet.post.utils import *

# from src.img2brep.brep.utils_old import *

data_root = r"E:\data\img2brep\failed_folder"
out_root = r"E:\data\img2brep\deepcad_whole_train_v4"
data_root = r"E:\data\img2brep\deepcad_whole_train_v4"

# data_root = r"H:\Data\img2brep"
# out_root = r"H:\Data\img2brep"

safe_check_dir(out_root)


def construct_brep_from_folder_path(folder_name, isdebug=False):
    # check if the rencon_brep.stl exists and clear
    safe_check_dir(os.path.join(out_root, folder_name))
    for file_name in os.listdir(os.path.join(out_root, folder_name)):
        if file_name.endswith(".stl"):
            os.remove(os.path.join(out_root, folder_name, file_name))

    data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'))

    # Face sample points (num_faces*20*20*3)
    face_points = data_npz['sample_points_faces'][:, ::1, ::1]
    line_points = data_npz['sample_points_lines'][:, ::1]
    vertex_points = data_npz['sample_points_vertices']
    face_edge_loop = data_npz['face_edge_loop']

    # add some noise to the points
    # face_points = face_points + np.random.normal(0, 1e-6, face_points.shape)
    # line_points = line_points + np.random.normal(0, 1e-3, line_points.shape)

    # print(f"face mse after add noise: {np.mean((face_points - data_npz['sample_points_faces']) ** 2)}")
    # print(f"line mse after add noise: {np.mean((line_points - data_npz['sample_points_lines']) ** 2)}")

    #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
    edge_face_connectivity = data_npz['edge_face_connectivity']
    #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
    vertex_edge_connectivity = data_npz['vertex_edge_connectivity']

    gt_faces = face_points
    gt_edges = line_points
    gt_vertices = vertex_points
    gt_edge_face_connectivity = edge_face_connectivity
    gt_vertex_edge_connectivity = vertex_edge_connectivity
    gt_face_edge_loop = []
    for loop in face_edge_loop:
        loop = loop[np.logical_and(loop != -1, loop != -2)]
        gt_face_edge_loop.append(list(loop))

    # create face_edge_adj which is the edges list of each faces, so that we can construct the edge loops for each face
    face_edge_adj = [[] for _ in range(gt_faces.shape[0])]
    for edge_face1_face2 in gt_edge_face_connectivity:
        edge, face1, face2 = edge_face1_face2
        if face1 == face2:
            continue
        assert edge not in face_edge_adj[face1]
        face_edge_adj[face1].append(edge)

    edge_vertex_adj = [[-1, -1] for _ in range(gt_edges.shape[0])]
    for vertex_edge1_edge2 in gt_vertex_edge_connectivity:
        vertex, edge1, egde2 = vertex_edge1_edge2
        # vertex = replace_vertex[vertex] if vertex in replace_vertex else vertex
        assert edge_vertex_adj[edge1][1] == -1
        assert edge_vertex_adj[egde2][0] == -1
        edge_vertex_adj[edge1][1] = vertex
        edge_vertex_adj[egde2][0] = vertex
    edge_vertex_adj = np.array(edge_vertex_adj, dtype=np.int32)

    if isdebug:
        debug_face_save_path = os.path.join(out_root, folder_name, "debug_face_loop")
        check_dir(debug_face_save_path)
        export_point_cloud(os.path.join(debug_face_save_path, 'face.ply'), gt_faces.reshape(-1, 3))
        export_point_cloud(os.path.join(debug_face_save_path, 'edge.ply'), gt_edges.reshape(-1, 3))
        for face_idx in range(len(gt_face_edge_loop)):
            for idx, edge_idx in enumerate(gt_face_edge_loop[face_idx]):
                export_point_cloud(
                        os.path.join(debug_face_save_path, f"face{face_idx}_edge{idx}_{edge_idx}_{edge_vertex_adj[edge_idx]}.ply"),
                        gt_edges[edge_idx],
                        np.linspace([1, 0, 0], [0, 1, 0], gt_edges[edge_idx].shape[0]))

    solid, faces_result = construct_brep(gt_faces, gt_edges, face_edge_adj, isdebug=isdebug,
                                         folder_path=os.path.join(out_root, folder_name))

    if solid.ShapeType() == TopAbs_COMPOUND:
        print(f"solid is TopAbs_COMPOUND {folder_name}")
        write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_compound.stl'), linear_deflection=0.1,
                       angular_deflection=0.5)
        return 2, faces_result

    safe_check_dir(os.path.join(out_root, folder_name))

    analyzer = BRepCheck_Analyzer(solid)

    if not analyzer.IsValid():
        # iterator = TopoDS_Iterator(solid)
        # while iterator.More():
        #     sub_shape = iterator.Value()
        #     status = analyzer.Result(sub_shape)
        #     if status != BRepCheck_Status.BRepCheck_NoError:
        #         sub_shape_type = sub_shape.ShapeType()
        #         print(f"Sub-shape type: {sub_shape_type}, Error: {status}")
        #         viz_shape(sub_shape)
        #     iterator.Next()
        write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_invaild.stl'), linear_deflection=0.1,
                       angular_deflection=0.5)

        return 3, faces_result

    try:
        if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.stl')):
            os.remove(os.path.join(out_root, folder_name, 'recon_brep.stl'))
        if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.step')):
            os.remove(os.path.join(out_root, folder_name, 'recon_brep.step'))
        write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
        write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'), linear_deflection=0.1,
                       angular_deflection=0.5)
    except:
        print(f"Failed to write brep for {folder_name}")
        return 4, faces_result
        pass

    if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.stl')):
        return 0, faces_result
    else:
        return -1, faces_result


# 00990947
# 00996077

# "00000443"
# "00000615"
# "00001971"
# “00005191”
# "00001971"
# "00000330"
# 00000443_failed_2
# 00150671_failed_2
# 00013487_fixed
# test_folder = "00000102_invalid_3"
# test_folder = "00000168_invalid_3"
# test_folder = "00000131_fixed"
# test_folder = "00000007_fixed"
# test_folder = "00000093_fixed"

if True:
    test_folder = "00000007_fixed"
    r, face_result = construct_brep_from_folder_path(test_folder, isdebug=True)
    assert sum(face_result) == len(face_result)
    exit(0)

all_folders = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
all_folders.sort()

all_folders = all_folders[200:1000]

folder2r_dict = {}
r_count_dict = {}

success_folders = []
for folder_name in tqdm.tqdm(all_folders):
    print("========== Beginning Processing", folder_name, " =========")
    # if folder_name.split("_")[1] == "fixed":
    #     r = 0
    #     continue

    try:
        r, _ = construct_brep_from_folder_path(folder_name)
    except Exception as e:
        with open(os.path.join(out_root, folder_name, "error.txt"), "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        r = -1

    r_count_dict[r] = r_count_dict.get(r, 0) + 1
    folder2r_dict[folder_name] = r

    if r == 0:
        os.rename(os.path.join(data_root, folder_name),
                  os.path.join(data_root, folder_name.split("_")[0] + "_fixed"))
    elif r == 3:
        os.rename(os.path.join(data_root, folder_name),
                  os.path.join(data_root, folder_name.split("_")[0] + "_invalid_" + str(r)))
    else:
        os.rename(os.path.join(data_root, folder_name),
                  os.path.join(data_root, folder_name.split("_")[0] + "_failed_" + str(r)))

    print("========== End Processing", r, " =========")
    pass

print(r_count_dict)
