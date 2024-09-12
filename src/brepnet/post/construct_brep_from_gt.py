import os
import shutil
import sys
import traceback
import ray

import numpy as np
import tqdm
from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_Status
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from OCC.Core.TopoDS import TopoDS_Iterator

# from pytorch3d.loss import chamfer_distance

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud

from src.brepnet.post.utils import *

v_data_root = r"/mnt/d/data/deepcad_whole_train_v5"
v_out_root = r"/mnt/d/data/deepcad_whole_train_v5_out"
safe_check_dir(v_out_root)


def construct_brep_from_data(data_root, out_root, folder_name_list, isdebug=False):
    for folder_name in folder_name_list:
        try:
            safe_check_dir(os.path.join(out_root, folder_name))
            if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.stl')):
                os.remove(os.path.join(out_root, folder_name, 'recon_brep.stl'))
            if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.step')):
                os.remove(os.path.join(out_root, folder_name, 'recon_brep.step'))

            data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'))

            # Face sample points (num_faces*20*20*3)
            face_points = data_npz['sample_points_faces']
            line_points = data_npz['sample_points_lines']

            #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
            edge_face_connectivity = data_npz['edge_face_connectivity']

            gt_faces = face_points
            gt_edges = line_points
            gt_edge_face_connectivity = edge_face_connectivity

            # create face_edge_adj which is the edges list of each faces, so that we can construct the edge loops for each face
            face_edge_adj = [[] for _ in range(gt_faces.shape[0])]
            for edge_face1_face2 in gt_edge_face_connectivity:
                edge, face1, face2 = edge_face1_face2
                if face1 == face2:
                    continue
                assert edge not in face_edge_adj[face1]
                face_edge_adj[face1].append(edge)

            if isdebug:
                debug_face_save_path = os.path.join(out_root, folder_name, "debug_face_loop")
                check_dir(debug_face_save_path)
                export_point_cloud(os.path.join(debug_face_save_path, 'face.ply'), gt_faces.reshape(-1, 3))
                export_point_cloud(os.path.join(debug_face_save_path, 'edge.ply'), gt_edges.reshape(-1, 3))
                for face_idx in range(len(face_edge_adj)):
                    for idx, edge_idx in enumerate(face_edge_adj[face_idx]):
                        export_point_cloud(
                                os.path.join(debug_face_save_path, f"face{face_idx}_edge{idx}_{edge_idx}.ply"),
                                gt_edges[edge_idx],
                                np.linspace([1, 0, 0], [0, 1, 0], gt_edges[edge_idx].shape[0]))

            solid, faces_result = construct_brep(gt_faces, gt_edges, face_edge_adj, isdebug=isdebug, is_save_face=True,
                                                 folder_path=os.path.join(out_root, folder_name))

            if solid.ShapeType() == TopAbs_COMPOUND:
                print(f"solid is TopAbs_COMPOUND {folder_name}")

            analyzer = BRepCheck_Analyzer(solid)
            if not analyzer.IsValid():
                print(f"solid is invalid {folder_name}")

            try:
                write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
                write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'), linear_deflection=0.01,
                               angular_deflection=0.5)
            except:
                print(f"Failed to write brep for {folder_name}")
        except Exception as e:
            with open(os.path.join(out_root, "error.txt"), "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(folder_name + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(folder_name + ": " + str(e))
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(e)
                shutil.rmtree(os.path.join(out_root, folder_name))


construct_brep_from_data_ray = ray.remote(construct_brep_from_data)


def test_():
    debug_folder = ["00005083"]
    construct_brep_from_data(v_data_root, v_out_root, debug_folder, isdebug=True)


if __name__ == '__main__':
    ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=15000,
            # num_cpus=1,
            # local_mode=True
    )

    all_folders = [folder for folder in os.listdir(v_data_root) if os.path.isdir(os.path.join(v_data_root, folder))]
    all_folders.sort()

    batch_size = 100
    num_batches = len(all_folders) // batch_size + 1

    tasks = []
    for i in range(num_batches):
        tasks.append(construct_brep_from_data_ray.remote(v_data_root, v_out_root, all_folders[i * batch_size:(i + 1) * batch_size]))

    ray.get(tasks)
    print("Done")
