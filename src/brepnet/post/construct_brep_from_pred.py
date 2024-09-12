import os.path
import random
import shutil
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

data_root = r"/mnt/d/img2brep/0909_test_export"
out_root = r"/mnt/d/img2brep/0909_test_export"
save_root = r"/mnt/d/img2brep/0909_test_export_success"

safe_check_dir(out_root)


def is_circle(points, tolerance=0.0001):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    radius = np.mean(distances)
    error = np.mean((distances - radius) ** 2)
    return error < tolerance


def construct_brep_from_folder_path(folder_name, isdebug=False):
    # check if the rencon_brep.stl exists and clear
    safe_check_dir(os.path.join(out_root, folder_name))
    for file_name in os.listdir(os.path.join(out_root, folder_name)):
        if file_name.endswith(".stl"):
            os.remove(os.path.join(out_root, folder_name, file_name))

    data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'), allow_pickle=True)
    pred_npz = np.load(os.path.join(data_root, folder_name, f'{folder_name}.npz'), allow_pickle=True)

    pred_npz = pred_npz['arr_0'].item()

    gt_egde_points = data_npz['sample_points_lines']

    # Face sample points (num_faces*20*20*3)
    recon_faces = pred_npz['pred_face']
    recon_edges = pred_npz['pred_edge']
    edge_face_connectivity = pred_npz['pred_edge_face_connectivity']

    # assert gt_egde_points.shape[0] == pred_npz['pred_edge'].shape[0]

    for idx in range(recon_edges.shape[0]):
        if is_circle(recon_edges[idx]):
            recon_edges[idx][-1] = recon_edges[idx][0]

    debug_face_save_path = os.path.join(out_root, folder_name, "debug_face_loop")
    check_dir(debug_face_save_path)
    export_point_cloud(os.path.join(debug_face_save_path, 'face.ply'), recon_faces.reshape(-1, 3))
    export_point_cloud(os.path.join(debug_face_save_path, 'edge.ply'), recon_edges.reshape(-1, 3))

    assert edge_face_connectivity.shape[0] == recon_edges.shape[0]

    # create face_edge_adj which is the edges list of each faces, so that we can construct the edge loops for each face
    face_edge_adj = [[] for _ in range(recon_faces.shape[0])]
    for edge_face1_face2 in edge_face_connectivity:
        edge, face1, face2 = edge_face1_face2
        if face1 == face2:
            raise
        assert edge not in face_edge_adj[face1]
        face_edge_adj[face1].append(edge)

    solid, faces_result = construct_brep(recon_faces, recon_edges, face_edge_adj, isdebug=isdebug,
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

# 00002718

# if True:
#     # 00007186
#     test_folder = "00007186"
#     r, face_result = construct_brep_from_folder_path(test_folder, isdebug=True)
#     # assert sum(face_result) == len(face_result)
#     exit(0)

all_folders = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
all_folders.sort()
random.shuffle(all_folders)

success_folders = []
exception_folders = []
os.makedirs(save_root, exist_ok=True)
for folder_name in tqdm.tqdm(all_folders):
    print("========== Beginning Processing", folder_name, " =========")
    try:
        r, _ = construct_brep_from_folder_path(folder_name)
        if r == 0:
            success_folders.append(folder_name)
            if not os.path.exists(os.path.join(save_root, folder_name)):
                shutil.copytree(os.path.join(data_root, folder_name), os.path.join(save_root, folder_name))
    except Exception as e:
        r = -1
        exception_folders.append(folder_name)
    print("========== End Processing", r, " =========")
    pass

print("Success Folders: ", success_folders)
print("Exception Folders: ", exception_folders)
