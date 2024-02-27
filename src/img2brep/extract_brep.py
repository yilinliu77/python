import os
import queue
from pathlib import Path

import ray, trimesh
import open3d as o3d
import numpy as np
import yaml
from tqdm import tqdm
from shared.common_utils import *

data_root = Path(r"G:/Dataset/ABC/raw_data/abc_0000_obj_v00")
data_split = r"valid_planar_shapes_except_cube.txt"
output_root = Path(r"G:/Projects/img2brep/data/0planar_shapes")

@ray.remote(num_cpus=1)
def get_brep(v_root, output_root, v_folders):
    for idx, v_folder in enumerate(v_folders):
    # for idx, v_folder in enumerate(tqdm(v_folders)):
        safe_check_dir(output_root / v_folder)

        try:
            all_files = os.listdir(v_root / v_folder)
            obj_file = [ff for ff in all_files if ff.endswith(".obj")][0]
            yml_file = [ff for ff in all_files if ff.endswith(".yml") and "features" in ff][0]

            mesh = trimesh.load_mesh(v_root / v_folder / obj_file, process=False, maintain_order=True)

            # Normalize with bounding box
            extent = mesh.bounding_box.extents
            diag = np.linalg.norm(extent)

            mesh.vertices -= mesh.bounding_box.centroid
            mesh.vertices /= diag

            mesh.export(output_root / v_folder / "mesh.ply")

            # Extract BREP
            files = yaml.load(open(v_root / v_folder / yml_file, "r").read(), yaml.CLoader)
            curves = files["curves"]
            surfaces = files["surfaces"]

            planes = []
            lines = []

            for line in curves:
                coords = np.asarray(line["location"])
                direction = np.asarray(line["direction"])
                vertex_index = set(line["vert_indices"])
                lines.append({"coords": coords, "direction": direction, "vertex_index": vertex_index})
            for plane in surfaces:
                params = np.asarray(plane["coefficients"])
                vertex_index = set(plane["vert_indices"])
                planes.append({"params": params, "vertex_index": vertex_index})

            # Recover the topologies

            # Plane-to-line connectivity
            plane_to_line = [[] for _ in range(len(planes))]
            for id_plane, plane in enumerate(planes):
                id_lines = []
                id_vertices = []
                for id_line, line in enumerate(lines):
                    id_common_points = plane["vertex_index"] & line["vertex_index"]
                    if len(id_common_points) < 2:
                        continue
                    id_lines.append(id_line)
                    id_vertices.append(id_common_points)

                sequences = []
                is_visited = [False] * len(id_lines)
                que = queue.Queue()
                for item in range(len(id_lines)):
                    if is_visited[item]:
                        continue
                    que.put(item)
                while not que.empty():
                    if min(is_visited):
                        break
                    id_start = que.get()
                    id_line1 = id_start
                    sequence = []
                    stop = False
                    while not stop:
                        stop = True
                        for id_line2 in range(len(id_lines)):
                            vertices2 = id_vertices[id_line2]
                            if id_line1 == id_line2 or is_visited[id_line2]:
                                continue
                            common_points = id_vertices[id_line1] & vertices2
                            if len(common_points) != 1:
                                continue
                            id_vertex = common_points.pop()
                            if id_vertex in sequence:
                                continue
                            stop = False
                            sequence.append(id_vertex)
                            id_line1 = id_line2
                            is_visited[id_line2] = True
                            if id_line2 == id_start:
                                stop = True
                            break
                    is_visited[id_start] = True
                    if len(sequence)>1:
                        sequences.append(sequence)
                plane_to_line[id_plane] = sequences

            with open(output_root / v_folder /"plane_to_line.txt", "w") as f:
                for plane in plane_to_line:
                    f.write("p\n")
                    for sequences in plane:
                        f.write(",".join([str(item) for item in sequences]) + "\n")

            # Check
            line_mesh = o3d.geometry.LineSet()
            points = []
            indices = []
            existing_vertices_id = []
            for plane in plane_to_line:
                # points = []
                for sequence in plane:
                    lines = zip(sequence, sequence[1:]+sequence[0:1])
                    for seg in lines:
                        indices.append([])
                        if seg[0] not in existing_vertices_id:
                            points.append(mesh.vertices[seg[0]])
                            existing_vertices_id.append(seg[0])
                        indices[-1].append(existing_vertices_id.index(seg[0]))

                        if seg[1] not in existing_vertices_id:
                            points.append(mesh.vertices[seg[1]])
                            existing_vertices_id.append(seg[1])
                        indices[-1].append(existing_vertices_id.index(seg[1]))

                        # points.append(mesh.vertices[seg[0]])
                        # points.append(mesh.vertices[seg[1]])
                # line_mesh.points = o3d.utility.Vector3dVector(points)
                # line_mesh.lines = o3d.utility.Vector2iVector(np.array([i for i in range(len(points))]).reshape(-1, 2))
                # o3d.io.write_line_set("{}_loops.ply".format(v_folder), line_mesh)
                continue

            line_mesh.points = o3d.utility.Vector3dVector(points)
            line_mesh.lines = o3d.utility.Vector2iVector(indices)
            o3d.io.write_line_set(str(output_root / v_folder / "wireframe.ply"), line_mesh)

            plane_adj = np.zeros((len(planes), len(planes)), dtype=np.int32)
            for id_plane1, plane1 in enumerate(planes):
                for id_plane2, plane2 in enumerate(planes):
                    if id_plane1 >= id_plane2:
                        continue
                    if len(plane1["vertex_index"] & plane2["vertex_index"]) != 0:
                        plane_adj[id_plane1, id_plane2] = 1
                        plane_adj[id_plane2, id_plane1] = 1

            with open(output_root / v_folder/ "plane_adj.txt", "w") as f:
                for row in plane_adj:
                    f.write(",".join([str(item) for item in row]) + "\n")

        except Exception as e:
            with open(output_root/"error.txt", "a") as f:
                f.write(v_folder + str(e) + "\n")

        continue

    return

if __name__ == '__main__':
    total_ids = [item.strip() for item in open(data_split, "r").readlines()]

    safe_check_dir(output_root)

    # single process
    if False:
        get_brep(data_root, output_root, total_ids)
    else:
        ray.init(
            # num_cpus=1,
            # local_mode=True
        )
        num_batches = 40
        batch_size = len(total_ids) // num_batches + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(
                get_brep.remote(
                    data_root,
                    output_root,
                    total_ids[i*batch_size:min(len(total_ids),(i+1)*batch_size)]))
        ray.get(tasks)
        print("Done")
