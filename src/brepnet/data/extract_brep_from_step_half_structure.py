import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm
import ray, trimesh
import numpy as np
from OCC.Core import TopoDS, TopExp, BRepBndLib
from OCC.Core.AIS import AIS_Shape, AIS_WireFrame
from OCC.Core.Approx import Approx_Curve3d
from OCC.Core.Aspect import Aspect_TypeOfLine
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Geom import Geom_BoundedCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.GeomLProp import GeomLProp_SLProps, GeomLProp_CLProps
from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial_Silver, Graphic3d_TypeOfShadingModel, \
    Graphic3d_NameOfMaterial_Brass, Graphic3d_TypeOfReflection, Graphic3d_AspectLine3d
from OCC.Core.Prs3d import Prs3d_LineAspect
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex, TopoDS_Wire
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_COMPOUND
from OCC.Core.GeomAbs import (GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_Ellipse,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_C1, GeomAbs_C2)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core._TopAbs import TopAbs_REVERSED
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Dir
from OCC.Display.OCCViewer import OffscreenRenderer, Viewer3d
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file, write_step_file
import traceback, sys

from PIL import Image

from shared.occ_utils import normalize_shape, get_triangulations, get_primitives, get_ordered_edges
from src.brepnet.post.utils import Shape

# from src.brepnet.post.utils import construct_solid

debug_id = None
# debug_id = "00000162"

write_debug_data = True
check_post_processing = True
# data_root = Path(r"/mnt/e/yilin/data_step")
# output_root = Path(r"/mnt/d/img2brep/abc_v2/")
# data_split = r"src/brepnet/data/list/abc_total.txt"

data_root = Path(r"E:\data\img2brep\train_brepquery\data_step")
output_root = Path(r"E:\data\img2brep\train_brepquery\abc_v3")
data_split = r"E:\data\img2brep\train_brepquery\abc_total.txt"

exception_files = [
    r"src/brepnet/bak/list_bak/abc_cube_ids.txt",
    r"src/brepnet/bak/list_bak/abc_with_others_ids.txt",
]

num_max_primitives = 100000
sample_resolution = 16

face_scale_threshold = 1e-3
edge_scale_threshold = 1e-3


def check_dir(v_path):
    if os.path.exists(v_path):
        shutil.rmtree(v_path)
    os.makedirs(v_path)


def safe_check_dir(v_path):
    if not os.path.exists(v_path):
        os.makedirs(v_path)


def diable_occ_log():
    from OCC.Core.Message import Message_Alarm, message
    printers = message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)


def get_brep(v_root, output_root, v_folder):
    random.seed(0)
    np.random.seed(0)
    safe_check_dir(output_root / v_folder)
    diable_occ_log()

    try:
        # Load mesh and yml files
        all_files = os.listdir(v_root / v_folder)
        step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff][0]

        # Start to extract BREP
        shape = read_step_file(str(v_root / v_folder / step_file), verbosity=False)
        if shape.NbChildren() != 1:
            raise ValueError("Multiple components: {}; Jump over".format(v_folder))

        shape = normalize_shape(shape, 0.9)
        write_step_file(shape, str(output_root / v_folder / "normalized_shape.step"))

        v, f = get_triangulations(shape, 0.001)
        mesh = trimesh.Trimesh(vertices=np.array(v), faces=np.array(f))
        mesh.export(output_root / v_folder / "mesh.ply")
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v), triangles=o3d.utility.Vector3iVector(f))
        pc = mesh.sample_points_poisson_disk(10000, use_triangle_normal=True)
        o3d.io.write_point_cloud(str(output_root / v_folder / "pc.ply"), pc)

        # Explore and list faces, edges, and vertices
        face_dict = {}
        for face in get_primitives(shape, TopAbs_FACE):
            if face not in face_dict:
                face_dict[face] = len(face_dict)
        num_faces = len(face_dict)

        # Store the adj faces of each edge
        edge_face_look_up_table = {}
        for face in face_dict:
            edges = get_ordered_edges(face)
            for wire in edges:
                for edge in wire:
                    if edge not in edge_face_look_up_table:
                        edge_face_look_up_table[edge] = [face]
                    else:
                        edge_face_look_up_table[edge].append(face)

        # For each intersection (face-face), store the corresponding edges
        face_face_adj_dict = {}  # Used to check if two face produce more than 1 edge
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

        edge_face_connectivity = []
        # Merge curve if more than 1 edge are produced
        for key in face_face_adj_dict:
            curves = [item for item in face_face_adj_dict[key]]
            if key[0] == key[1]:  # Skip seam line
                continue

            if len(curves) == 1:
                curve = curves[0]
            else:
                edges_seq = TopTools_HSequenceOfShape()
                for edge in curves:
                    edges_seq.Append(edge)
                wire_array_c = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, 0.001, False)
                if wire_array_c.Length() != 1:
                    raise Exception("Error: Wire creation failed")
                wire = TopoDS.topods.Wire(wire_array_c.First())

                sample_points = []
                wire_explorer = BRepTools_WireExplorer(wire)
                while wire_explorer.More():
                    edge = TopoDS.topods.Edge(wire_explorer.Current())
                    curve = BRepAdaptor_Curve(edge)
                    range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
                    range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
                    sample_u = np.linspace(range_start, range_end, num=sample_resolution)
                    for u in sample_u:
                        sample_points.append(curve.Value(u))
                    wire_explorer.Next()
                # Fit BSpline
                u_points_array = TColgp_Array1OfPnt(1, len(sample_points))
                for i in range(len(sample_points)):
                    u_points_array.SetValue(i + 1, sample_points[i])
                curve = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 1e-3).Curve()
                curve = BRepBuilderAPI_MakeEdge(curve).Edge()

            edge_face_connectivity.append((
                curve,
                key[0],
                key[1]
            ))

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
                    props = GeomLProp_SLProps(surface.Surface().Surface(), u[i, j], v[i, j], 1, 0.01)
                    dir = props.Normal()
                    points.append(np.array([pnt.X(), pnt.Y(), pnt.Z(), dir.X(), dir.Y(), dir.Z()], dtype=np.float32))

            # Check face range
            points = np.stack(points, axis=0)
            face_center = np.mean(points[:, :3], axis=0)
            face_scale = np.min(np.linalg.norm(points[:, :3] - face_center, axis=1))
            if face_scale < face_scale_threshold:
                raise ValueError("Face scale too small: {}".format(face_scale))
            face_sample_points.append(points.reshape(sample_resolution, sample_resolution, -1))
        face_sample_points = np.stack(face_sample_points, axis=0)
        assert len(face_dict) == num_faces == face_sample_points.shape[0]

        # Sample points in edges
        edge_sample_points = []
        num_edges = len(edge_face_connectivity)
        for i_intersection in range(num_edges):
            edge, id_face1, id_face2 = edge_face_connectivity[i_intersection]
            edge_face_connectivity[i_intersection] = (i_intersection, id_face1, id_face2)
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve]:
                raise ValueError("Unsupported curve type: {}".format(curve.GetType()))
            # Sample 20 points along it
            # Determine the orientation
            range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
            range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
            sample_u = np.linspace(range_start, range_end, num=sample_resolution)
            points = []
            for u in sample_u:
                pnt = curve.Value(u)
                v1 = gp_Vec()
                curve.D1(u, pnt, v1)
                v1 = v1.Normalized()
                points.append(np.array([pnt.X(), pnt.Y(), pnt.Z(), v1.X(), v1.Y(), v1.Z()], dtype=np.float32))

            # Check edge range
            points = np.stack(points, axis=0)
            edge_center = np.mean(points[:, :3], axis=0)
            edge_scale = np.min(np.linalg.norm(points[:, :3] - edge_center, axis=1))
            if edge_scale < edge_scale_threshold:
                raise ValueError("Edge scale too small: {}".format(edge_scale))
            edge_sample_points.append(np.stack(points, axis=0))
        edge_sample_points = np.stack(edge_sample_points, axis=0)
        edge_face_connectivity = np.asarray(edge_face_connectivity, dtype=np.int32)

        # Accelerate the training
        # Prepare edge face intersection data
        face_adj = np.zeros((num_faces, num_faces), dtype=bool)
        face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True

        zero_positions = np.stack(np.where(face_adj == 0), axis=1)

        data_dict = {
            'sample_points_lines'   : edge_sample_points.astype(np.float32),
            'sample_points_faces'   : face_sample_points.astype(np.float32),

            'edge_face_connectivity': edge_face_connectivity.astype(np.int64),

            "face_adj"              : face_adj,
            "zero_positions"        : zero_positions,
        }

        np.savez_compressed(output_root / v_folder / "data.npz", **data_dict)

        if check_post_processing:
            from src.brepnet.post.construct_brep import construct_brep_from_datanpz

            construct_brep_from_datanpz(str(output_root),
                                        str(output_root),
                                        v_folder,
                                        v_drop_num=0,
                                        use_cuda=True, from_scratch=False,
                                        is_save_data=False, is_log=False,
                                        is_optimize_geom=False, v_max_optimize_iter=200,
                                        is_ray=False, )

            # from src.brepnet.post.utils import construct_brep
            #
            # shape = Shape(face_sample_points[...,:3], edge_sample_points[...,:3], edge_face_connectivity, False)
            # shape.remove_half_edges()  # will filter some invalid intersection edges
            # shape.check_openness()
            # shape.build_fe()
            # shape.build_vertices(0.2)
            # _, mesh, solid = construct_brep(
            #         shape,
            #         2e-2,
            #         False,
            #         # debug_face_idx=[4]
            # )
            # if solid is None:
            #     raise ValueError("Post processing failed")
            # write_step_file(solid, str(output_root / v_folder / "post_processed_shape.step"))
            pass

        if write_debug_data:
            import open3d as o3d
            pc_model = o3d.geometry.PointCloud()
            pc_model.points = o3d.utility.Vector3dVector(face_sample_points[..., :3].reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "face_sample_points.ply"), pc_model)

            pc_model.points = o3d.utility.Vector3dVector(edge_sample_points[..., :3].reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "edge_sample_points.ply"), pc_model)
            pass

    except Exception as e:
        with open(output_root / "error.txt", "a") as f:
            tb_list = traceback.extract_tb(sys.exc_info()[2])
            last_traceback = tb_list[-1]
            f.write(v_folder + ": " + str(e) + "\n")
            f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
            # print(v_folder + ": " + str(e))
            # print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
            shutil.rmtree(output_root / v_folder)
        return False
    return True


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
    # total_ids = total_ids[:100]
    print("Total ids: {} -> {}".format(num_original, len(total_ids)))
    check_dir(output_root)

    # single process
    if debug_id is not None:
        total_ids = debug_id
        get_brep(data_root, output_root, total_ids)
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=15000,
                # num_cpus=1,
                # local_mode=True
        )
        tasks = []
        for i in range(len(total_ids)):
            tasks.append(get_brep_ray.remote(data_root, output_root, total_ids[i]))
        results = []
        for i in tqdm(range(len(total_ids))):
            results.append(ray.get(tasks[i]))
        print("Processed {}/{} files".format(sum(results), len(total_ids)))
