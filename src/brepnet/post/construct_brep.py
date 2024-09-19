import os, sys, shutil, traceback, tqdm

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud

from src.brepnet.post.utils import *
from src.brepnet.post.geom_optimization import optimize_geom, test_optimize_geom

import ray
import argparse


def construct_brep_from_datanpz(data_root, out_root, folder_name_list, is_optimize_geom=False, isdebug=False):
    for folder_name in folder_name_list:
        try:
            safe_check_dir(os.path.join(out_root, folder_name))
            if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.stl')):
                os.remove(os.path.join(out_root, folder_name, 'recon_brep.stl'))
            if os.path.exists(os.path.join(out_root, folder_name, 'recon_brep.step')):
                os.remove(os.path.join(out_root, folder_name, 'recon_brep.step'))

            # specify the key to get the face points, edge points and edge_face_connectivity in data.npz
            data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'), allow_pickle=True)['arr_0'].item()
            if 'sample_points_faces' in data_npz:
                face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
                edge_points = data_npz['sample_points_lines']  # Edge sample points (num_lines*20*3)
                edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
            elif 'pred_face' in data_npz:
                face_points = data_npz['pred_face']
                edge_points = data_npz['pred_edge']
                edge_face_connectivity = data_npz['pred_edge_face_connectivity']
            else:
                raise ValueError(f"Unknown data npz format {folder_name}")

            # face_edge_adj store the edge idx list of each face
            face_edge_adj = [[] for _ in range(face_points.shape[0])]
            for edge_face1_face2 in edge_face_connectivity:
                edge, face1, face2 = edge_face1_face2
                if face1 == face2:
                    raise ValueError(f"face1 == face2 {folder_name}")
                assert edge not in face_edge_adj[face1]
                face_edge_adj[face1].append(edge)

            if isdebug:
                debug_face_save_path = str(os.path.join(out_root, folder_name, "debug_face_loop"))
                check_dir(debug_face_save_path)
                export_point_cloud(os.path.join(debug_face_save_path, 'face.ply'), face_points.reshape(-1, 3))
                export_point_cloud(os.path.join(debug_face_save_path, 'edge.ply'), edge_points.reshape(-1, 3))
                for face_idx in range(len(face_edge_adj)):
                    for idx, edge_idx in enumerate(face_edge_adj[face_idx]):
                        export_point_cloud(os.path.join(debug_face_save_path, f"face{face_idx}_edge{idx}_{edge_idx}.ply"),
                                           edge_points[edge_idx].reshape(-1, 3),
                                           np.linspace([1, 0, 0], [0, 1, 0], edge_points[edge_idx].shape[0]))

            # face_points = face_points + np.random.normal(0, 1e-3, size=(face_points.shape[0], 1, 1, 1))
            # edge_points = edge_points + np.random.normal(0, 1e-3, size=(edge_points.shape[0], 1, 1))

            if is_optimize_geom:
                face_points, edge_points = optimize_geom(face_points, edge_points, edge_face_connectivity, face_edge_adj)

                if isdebug:
                    debug_face_save_path = str(os.path.join(out_root, folder_name, "debug_face_loop"))
                    safe_check_dir(debug_face_save_path)
                    export_point_cloud(os.path.join(debug_face_save_path, 'optimized_face.ply'), face_points.reshape(-1, 3))
                    export_point_cloud(os.path.join(debug_face_save_path, 'optimized_edge.ply'), edge_points.reshape(-1, 3))

            # Construct Brep from face_points, edge_points, face_edge_adj
            solid, faces_result = construct_brep(face_points, edge_points, face_edge_adj,
                                                 isdebug=isdebug, is_save_face=True,
                                                 folder_path=os.path.join(out_root, folder_name))

            if solid.ShapeType() == TopAbs_COMPOUND:
                print(f"solid is TopAbs_COMPOUND {folder_name}")
                write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_compound.stl'),
                               linear_deflection=0.01, angular_deflection=0.5)
                continue

            analyzer = BRepCheck_Analyzer(solid)
            if not analyzer.IsValid():
                print(f"solid is invalid {folder_name}")
                write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_invalid.stl'),
                               linear_deflection=0.01, angular_deflection=0.5)
                continue

            # Valid Solid
            write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
            write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'), linear_deflection=0.01, angular_deflection=0.5)

        except Exception as e:
            with open(os.path.join(out_root, "error.txt"), "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(folder_name + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(folder_name + ": " + str(e))
                print(e)
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                shutil.rmtree(os.path.join(out_root, folder_name))


construct_brep_from_datanpz_ray = ray.remote(construct_brep_from_datanpz)


def test_construct_brep(v_data_root, v_out_root):
    debug_folder = ["00072639"]
    construct_brep_from_datanpz(v_data_root, v_out_root, debug_folder, is_optimize_geom=True, isdebug=True)
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, default=r"E:\data\img2brep\0916_context_test")
    parser.add_argument('--out_root', type=str, default=r"E:\data\img2brep\0916_context_test_out")
    args = parser.parse_args()
    v_data_root = args.data_root
    v_out_root = args.out_root
    safe_check_dir(v_out_root)
    if not os.path.exists(v_data_root):
        raise ValueError(f"Data root path {v_data_root} does not exist.")

    # test_construct_brep(v_data_root, v_out_root)

    ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            # num_cpus=1,
            # local_mode=True
    )

    all_folders = [folder for folder in os.listdir(v_data_root) if os.path.isdir(os.path.join(v_data_root, folder))]
    all_folders.sort()

    batch_size = 100
    num_batches = len(all_folders) // batch_size + 1

    tasks = []
    for i in range(num_batches):
        tasks.append(construct_brep_from_datanpz_ray.remote(v_data_root, v_out_root, all_folders[i * batch_size:(i + 1) * batch_size]))

    ray.get(tasks)
    print("Done")
