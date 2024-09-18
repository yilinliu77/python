import os
import shutil
from pathlib import Path

import ray, trimesh
import numpy as np
from OCC.Core import TopoDS, TopExp, BRepBndLib
from OCC.Core.Approx import Approx_Curve3d
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeWire
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Geom import Geom_BoundedCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex, TopoDS_Wire
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.GeomAbs import (GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_Ellipse,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_C1, GeomAbs_C2)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core._TopAbs import TopAbs_REVERSED
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Extend.DataExchange import read_step_file, write_step_file
import traceback, sys

write_debug_data = False
data_root = Path(r"d://data")
output_root = Path(r"d://data/deepcad_whole_train_v5")
data_split = r"src/img2brep/data/deepcad_train_whole.txt"

exception_files = [
    r"src/img2brep/data/abc_multiple_component_or_few_faces_ids_.txt",
    r"src/img2brep/data/abc_cube_ids.txt",
    r"src/img2brep/data/abc_with_others_ids.txt",
]

num_max_primitives = 100000
sample_resolution = 32

def check_dir(v_path):
    if os.path.exists(v_path):
        shutil.rmtree(v_path)
    os.makedirs(v_path)


def safe_check_dir(v_path):
    if not os.path.exists(v_path):
        os.makedirs(v_path)


# @ray.remote(num_cpus=1)
def get_brep(v_root, output_root, v_folders):
    # v_folders = ["00001000"]
    single_loop_folder = []

    for idx, v_folder in enumerate(v_folders):
        # for idx, v_folder in enumerate(tqdm(v_folders)):
        safe_check_dir(output_root / v_folder)

        try:
            # Load mesh and yml files
            all_files = os.listdir(v_root / v_folder)
            step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff][0]

            # Start to extract BREP
            shape = read_step_file(str(v_root / v_folder / step_file), verbosity=False)
            if shape.NbChildren() != 1:
                raise ValueError("Multiple components: {}; Jump over".format(v_folder))

            # Compute bounding box
            boundings = 0.9
            boundingBox = Bnd_Box()
            BRepBndLib.brepbndlib.Add(shape, boundingBox)
            xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()
            scale_x = boundings * 2 / (xmax - xmin)
            scale_y = boundings * 2 / (ymax - ymin)
            scale_z = boundings * 2 / (zmax - zmin)
            scaleFactor = min(scale_x, scale_y, scale_z)
            translation1 = gp_Vec(-(xmax + xmin) / 2, -(ymax + ymin) / 2, -(zmax + zmin) / 2)
            trsf1 = gp_Trsf()
            trsf1.SetTranslationPart(translation1)
            trsf2 = gp_Trsf()
            trsf2.SetScaleFactor(scaleFactor)
            trsf2.Multiply(trsf1)

            transformer = BRepBuilderAPI_Transform(trsf2)
            transformer.Perform(shape)
            shape = transformer.Shape()
            # Write step
            write_step_file(shape, str(output_root / v_folder / "normalized_shape.step"))

            mesh = BRepMesh_IncrementalMesh(shape, 0.001)
            # Get vertices and faces
            v = []
            f = []
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = face_explorer.Current()
                loc = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, loc)
                cur_vertex_size = len(v)
                for i in range(1, triangulation.NbNodes() + 1):
                    pnt = triangulation.Node(i)
                    v.append([pnt.X(), pnt.Y(), pnt.Z()])
                for i in range(1, triangulation.NbTriangles() + 1):
                    t = triangulation.Triangle(i)
                    if face.Orientation() == TopAbs_REVERSED:
                        f.append([t.Value(3) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1, t.Value(1) + cur_vertex_size - 1])
                    else:
                        f.append([t.Value(1) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1, t.Value(3) + cur_vertex_size - 1])
                face_explorer.Next()
            trimesh.Trimesh(vertices=np.array(v), faces=np.array(f)).export(output_root / v_folder / "mesh.ply")

            # Function to explore and print the elements of a shape
            def explore_shape(shape, shape_type):
                explorer = TopExp_Explorer(shape, shape_type)
                while explorer.More():
                    yield explorer.Current()
                    explorer.Next()

            # Explore and list faces, edges, and vertices
            face_dict = {}
            for face in explore_shape(shape, TopAbs_FACE):
                if face not in face_dict:
                    face_dict[face] = len(face_dict)
            num_faces = len(face_dict)
            edge_dict = {}
            for edge in explore_shape(shape, TopAbs_EDGE):
                if edge not in edge_dict:
                    edge_dict[edge] = len(edge_dict)
            num_edges = len(edge_dict)
            vertex_dict = {}
            for vertex in explore_shape(shape, TopAbs_VERTEX):
                if vertex not in vertex_dict and vertex.Reversed() not in vertex_dict:
                    vertex_dict[vertex] = len(vertex_dict)

            # Sample points in face
            face_sample_points = []
            for face in face_dict:
                surface = BRepAdaptor_Surface(face)

                if surface.GetType() not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                                             GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface]:
                    raise ValueError("Unsupported surface type: {}".format(surface.GetType()))

                first_u = surface.FirstUParameter()
                last_u = surface.LastUParameter()
                first_v = surface.FirstVParameter()
                last_v = surface.LastVParameter()
                u = np.linspace(first_u, last_u, num=sample_resolution)
                v = np.linspace(first_v, last_v, num=sample_resolution)
                u, v = np.meshgrid(u, v)
                points = []
                for i in range(u.shape[0]):
                    for j in range(u.shape[1]):
                        pnt = surface.Value(u[i, j], v[i, j])
                        points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
                face_sample_points.append(np.stack(points, axis=0).reshape(sample_resolution, sample_resolution, 3))
            face_sample_points = np.stack(face_sample_points, axis=0)
            assert len(face_dict) == num_faces == face_sample_points.shape[0]

            # Sample points in edges
            edge_sample_points = []
            for edge in edge_dict:
                curve = BRepAdaptor_Curve(edge)
                if curve.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve]:
                    raise ValueError("Unsupported curve type: {}".format(curve.GetType()))
                # Sample 20 points along it
                # Determine the orientation
                range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
                range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
                sample_u = np.linspace(range_start, range_end, num=sample_resolution)
                sample_points = []
                for u in sample_u:
                    pnt = curve.Value(u)
                    sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
                edge_sample_points.append(np.stack(sample_points, axis=0))
            edge_sample_points = np.stack(edge_sample_points, axis=0)
            assert len(edge_dict) == num_edges == edge_sample_points.shape[0]

            # Sample points in vertices
            vertex_sample_points = []
            for vertex in vertex_dict:
                pnt = BRep_Tool.Pnt(vertex)
                vertex_sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
            vertex_sample_points = np.stack(vertex_sample_points, axis=0)

            # (N, X) shows every loop with each face, -2 denotes a start token and pad with -1
            face_edge_loop = []
            vertex_edge_connectivity = []
            #
            edge_face_look_up_table = {}
            vertex_edge_look_up_table = {}
            for face in face_dict:
                loops = []
                for wire in explore_shape(face, TopAbs_WIRE):
                    loops.append(-2)
                    local_vertex_edge_connectivity = []
                    wire_explorer = BRepTools_WireExplorer(wire)

                    while wire_explorer.More():
                        edge = TopoDS.topods.Edge(wire_explorer.Current())
                        wire_explorer.Next()
                        if edge not in edge_dict:
                            raise ValueError("Edge not in edge_dict")
                        loops.append(edge_dict[edge])

                        if edge not in edge_face_look_up_table:
                            edge_face_look_up_table[edge] = [face]
                        else:
                            edge_face_look_up_table[edge].append(face)

                        second_vertex = TopExp.topexp.LastVertex(edge, True)
                        local_vertex_edge_connectivity.append((
                            vertex_dict[second_vertex] if second_vertex in vertex_dict else vertex_dict[
                                second_vertex.Reversed()], edge_dict[edge]))

                    for idv in range(len(local_vertex_edge_connectivity)):
                        vertex_edge_connectivity.append((
                            local_vertex_edge_connectivity[idv][0],
                            local_vertex_edge_connectivity[idv][1],
                            local_vertex_edge_connectivity[(idv + 1) % len(local_vertex_edge_connectivity)][1]
                        ))

                face_edge_loop.append(loops)

            face_face_adj_dict = {} # Used to check if two face produce more than 1 edge
            for edge in edge_face_look_up_table:
                if edge.Reversed() not in edge_face_look_up_table:
                    raise ValueError("Edge not in edge_face_look_up_table")
                if len(edge_face_look_up_table[edge]) != 1:
                    raise ValueError("Edge indexed by more than 1 faces.")

                item = (
                        face_dict[edge_face_look_up_table[edge][0]],
                        face_dict[edge_face_look_up_table[edge.Reversed()][0]]
                )

                if item not in face_face_adj_dict:
                    face_face_adj_dict[item] = [edge]
                else:
                    face_face_adj_dict[item].append(edge)
                #
                # edge_face_connectivity.append((
                #     edge_dict[edge],
                #     face_dict[edge_face_look_up_table[edge][0]],
                #     face_dict[edge_face_look_up_table[edge.Reversed()][0]]
                # ))

            edge_face_connectivity = []
            # Merge curve if more than 1 edge are produced
            for key in face_face_adj_dict:
                curves = [item for item in face_face_adj_dict[key]]
                if len(curves) == 1:
                    edge_face_connectivity.append((
                        edge_dict[curves[0]],
                        key[0],
                        key[1]
                    ))
                else:
                    wire_builder = BRepBuilderAPI_MakeWire()
                    for edge in curves:
                        wire_builder.Add(edge)
                    if not wire_builder.IsDone():
                        raise Exception("Error: Wire creation failed")
                    wire = wire_builder.Wire()

                    sample_points = []
                    wire_explorer = BRepTools_WireExplorer(wire)
                    while wire_explorer.More():
                        edge = TopoDS.topods.Edge(wire_explorer.Current())
                        curve = BRepAdaptor_Curve(edge)
                        range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
                        range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
                        sample_u = np.linspace(range_start, range_end, num=sample_resolution)
                        for u in sample_u:
                            pnt = curve.Value(u)
                            sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
                        wire_explorer.Next()
                    # Fit BSpline
                    u_points_array = TColgp_Array1OfPnt(1, len(sample_points))
                    for i in range(len(sample_points)):
                        u_points_array.SetValue(i + 1, gp_Pnt(*sample_points[i]))
                    curve = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 1e-3).Curve()
                    pass

            # Check
            if len(edge_face_connectivity) != len(edge_dict):
                raise ValueError("Wrong edge_face_connectivity")
            # if len(vertex_edge_connectivity) != num_edges / 2 * (num_edges / 2 - 1):
            #     raise ValueError("Wrong vertex_edge_connectivity")

            edge_face_connectivity = np.asarray(edge_face_connectivity, dtype=np.int32)
            vertex_edge_connectivity = np.asarray(vertex_edge_connectivity, dtype=np.int32)

            max_length = max(len(lst) for lst in face_edge_loop)
            face_edge_loop = np.array(
                [i + [-1] * (max_length - len(i)) for i in face_edge_loop], dtype=np.int32)

            # Accelerate the training
            # Prepare edge face intersection data
            face_adj = np.zeros((num_faces, num_faces), dtype=bool)
            face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True

            zero_positions = np.stack(np.where(face_adj == 0), axis=1)

            data_dict = {
                'sample_points_vertices': vertex_sample_points.astype(np.float32),
                'sample_points_lines': edge_sample_points.astype(np.float32),
                'sample_points_faces': face_sample_points.astype(np.float32),

                'face_edge_loop': face_edge_loop.astype(np.int64),
                'edge_face_connectivity': edge_face_connectivity.astype(np.int64),
                'vertex_edge_connectivity': vertex_edge_connectivity.astype(np.int64),

                "face_adj": face_adj,
                "zero_positions": zero_positions,
            }

            np.savez_compressed(output_root / v_folder / "data.npz", **data_dict)
            # continue
            if not write_debug_data:
                continue

            import open3d as o3d
            # Check
            # Write face
            pc_model = o3d.geometry.PointCloud()
            pc_model.points = o3d.utility.Vector3dVector(face_sample_points.reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "face_sample_points.ply"), pc_model)

            pc_model.points = o3d.utility.Vector3dVector(edge_sample_points.reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "edge_sample_points.ply"), pc_model)

            check_dir(output_root / v_folder / "debug_topology")
            for i in range(face_edge_loop.shape[0]):
                points = []
                for item in face_edge_loop[i]:
                    if item == -1 or item == -2:
                        continue
                    points.append(edge_sample_points[item])
                pc_model.points = o3d.utility.Vector3dVector(np.stack(points, axis=0).reshape(-1, 3))
                o3d.io.write_point_cloud(str(output_root / v_folder / "debug_topology" / "{}.ply".format(i)), pc_model)
        except Exception as e:
            with open(output_root / "error.txt", "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(v_folder + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(v_folder + ": " + str(e))
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(e)
                shutil.rmtree(output_root / v_folder)

    for folder in single_loop_folder:
        with open(output_root / "single_loop.txt", "a") as f:
            f.write(folder + "\n")

    return


get_brep_ray = ray.remote(get_brep)

if __name__ == '__main__':
    total_ids = [item.strip() for item in open(data_split, "r").readlines()]

    exception_ids = []
    for file in exception_files:
        exception_ids += [item.strip() for item in open(file, "r").readlines()]
    exception_ids = list(set(exception_ids))

    num_original = len(total_ids)
    total_ids = list(set(total_ids) - set(exception_ids))
    total_ids.sort()
    total_ids=["00005083"]
    print("Total ids: {} -> {}".format(num_original, len(total_ids)))
    check_dir(output_root)

    # single process
    if False:
        get_brep(data_root, output_root, total_ids)
    else:
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=15000,
            num_cpus=1,
            local_mode=True
        )
        batch_size = 100
        num_batches = len(total_ids) // batch_size + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(
                get_brep_ray.remote(data_root,
                                    output_root,
                                    total_ids[i * batch_size:min(len(total_ids), (i + 1) * batch_size)]))
        ray.get(tasks)
        print("Done")
