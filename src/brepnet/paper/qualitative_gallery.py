import os
import shutil
import sys
from OCC.Core import STEPControl
import OCC.Core.TopAbs as TopAbs
import OCC.Core.TopExp as TopExp
import OCC.Core.TopoDS as TopoDS
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
import trimesh
import tempfile
import sys
from pathlib import Path
import numpy as np

import trimesh.exchange
import trimesh.exchange.obj

from tqdm import tqdm

from shared.occ_utils import get_triangulations, get_primitives
from src.brepnet.eval.check_valid import check_step_valid_soild


def import_step_file_as_obj(file_path, step_edge_sample=16):
    valid, shape = check_step_valid_soild(file_path / "recon_brep.step", return_shape=True)

    v, f = get_triangulations(shape)
    mesh = trimesh.Trimesh(vertices=np.array(v), faces=np.array(f))
    vertices = np.asarray(mesh.vertices)
    bbox = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    scale = np.max(bbox[1] - bbox[0]) * 1.1
    center = np.mean(bbox, axis=0)
    v = np.asarray(mesh.vertices)
    rot_matrix = np.identity(4)
    if (bbox[1] - bbox[0]).argmin() == 0:
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], center)
        mesh.apply_transform(rot_matrix)
    v = (v - center) / scale
    mesh.vertices = np.dot(v, rot_matrix[:3, :3].T)
    mesh.fix_normals()  # Recalculates normals

    # Wire
    edges = get_primitives(shape, TopAbs.TopAbs_EDGE, True)
    v_wire = []
    l_wire = []
    vertex_index = 1
    for edge in edges:
        curve_adaptor = BRepAdaptor_Curve(edge)
        first = curve_adaptor.FirstParameter()
        last = curve_adaptor.LastParameter()
        for i in range(step_edge_sample):
            u = first + (last - first) * (i / (step_edge_sample - 1))  # Parameter for sampling
            p = gp_Pnt()
            curve_adaptor.D0(u, p)
            p = np.asarray([p.X(), p.Y(), p.Z()])
            v_wire.append(p)
        for i in range(vertex_index, vertex_index + step_edge_sample - 1):
            l_wire.append([i, i + 1])
        vertex_index += step_edge_sample
    v_wire = np.asarray(v_wire, dtype=np.float32)
    v_wire = (v_wire - center) / scale
    v_wire = np.dot(v_wire, rot_matrix[:3, :3].T)
    l_wire = np.asarray(l_wire, dtype=np.int32)
    # Vertex
    vertices = get_primitives(shape, TopAbs.TopAbs_VERTEX, True)
    v_vertex = [np.array([BRep_Tool.Pnt(vertex).X(), BRep_Tool.Pnt(vertex).Y(), BRep_Tool.Pnt(vertex).Z()]) for vertex
                in vertices]
    v_vertex = np.asarray(v_vertex, dtype=np.float32)
    v_vertex = (v_vertex - center) / scale
    v_vertex = np.dot(v_vertex, rot_matrix[:3, :3].T)

    min_z = np.asarray(mesh.vertices).min(axis=0)[2]
    mesh.vertices = mesh.vertices - np.array([0, 0, min_z])
    v_wire = v_wire - np.array([0, 0, min_z])
    v_vertex = v_vertex - np.array([0, 0, min_z])
    return mesh, (v_wire, l_wire), v_vertex


def import_model(v_path):
    prefix = v_path.stem
    mesh = trimesh.load_mesh(v_path / "{}_model.obj".format(prefix))

    vertices = np.asarray(mesh.vertices)
    bbox = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    scale = np.max(bbox[1] - bbox[0]) * 1.1
    center = np.mean(bbox, axis=0)

    rot_matrix = np.identity(4)
    if (bbox[1] - bbox[0]).argmin() == 0:
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], center)
        mesh.apply_transform(rot_matrix)

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale)
    mesh.fix_normals()  # Recalculates normals

    lv = np.asarray([list(map(lambda item: float(item), item.strip().split(" ")[1:])) for item in open(
        v_path / "{}_wire.obj".format(prefix)).readlines() if item[0] == "v"])
    ll = np.asarray([list(map(lambda item: int(item), item.strip().split(" ")[1:])) for item in open(
        v_path / "{}_wire.obj".format(prefix)).readlines() if item[0] == "l"])
    lv = (lv - center) / scale
    lv = np.dot(lv, rot_matrix[:3, :3].T)

    # Vertex
    vertices = np.asarray([list(map(lambda item: float(item), item.strip().split(" ")[1:])) for item in open(
        v_path / "{}_vertex.obj".format(prefix)).readlines() if item[0] == "v"])
    vertices = (vertices - center) / scale
    vertices = np.dot(vertices, rot_matrix[:3, :3].T)

    min_z = np.asarray(mesh.vertices).min(axis=0)[2]
    mesh.vertices = mesh.vertices - np.array([0, 0, min_z])
    lv = lv - np.array([0, 0, min_z])
    vertices = vertices - np.array([0, 0, min_z])

    return mesh, (lv, ll), vertices


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src.brepnet.paper.qualitative_gallery <succ_folder> <fail_folder>")
        sys.exit(1)
    succ_folder = Path(sys.argv[1])
    fail_folder = None
    if len(sys.argv) > 2:
        fail_folder = Path(sys.argv[2])

    out_folder = succ_folder.parent / "merged"
    out_folder.mkdir(exist_ok=True)

    # # Qualitative
    # num_row = 23
    # num_col = 17
    # x_offset = 1.5
    # y_offset = 2.0
    # Comparison
    num_row = 5
    num_col = 15
    x_offset = 1.5
    y_offset = 2.0

    np.random.seed(0)
    folders = sorted(succ_folder.iterdir())
    if fail_folder is not None:
        folders += sorted(fail_folder.iterdir())
    num_total_shapes = len(folders)
    if num_total_shapes > num_row * num_col:
        selected_index = np.random.choice(num_total_shapes, num_row * num_col, replace=False)
        selected_index.sort()
        folders = [folders[i] for i in selected_index]
    mesh_model = trimesh.Trimesh()
    wire_vertex = []
    wire_line = []
    i_wire_vertex = 0
    vertex_vertex = []

    for idx, prefix in enumerate(tqdm(folders)):
        # mesh_item, (v_wire_item, l_wire_item), v_vertex_item = import_step_file_as_obj(prefix, 100)
        mesh_item, (v_wire_item, l_wire_item), v_vertex_item = import_model(prefix)

        delta_x = idx % num_col * x_offset
        delta_y = idx // num_col * y_offset
        mesh_item.apply_translation([delta_x, delta_y, 0])
        v_wire_item += np.array([delta_x, delta_y, 0])
        v_vertex_item += np.array([delta_x, delta_y, 0])

        trimesh.PointCloud(v_vertex_item).export(out_folder / f"vertex_{idx}.obj")
        with open(out_folder / f"wire_{idx}.obj", "w") as f:
            for v in v_wire_item:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for l in l_wire_item:
                f.write(f"l {l[0]} {l[1]}\n")
        mesh_item.export(out_folder / f"model_{idx}.obj")

        mesh_model = trimesh.util.concatenate(mesh_model, mesh_item)
        wire_vertex.append(v_wire_item)
        l_wire_item += i_wire_vertex
        wire_line.append(l_wire_item)
        vertex_vertex.append(v_vertex_item)
        i_wire_vertex += v_wire_item.shape[0]
        
    wire_vertex = np.vstack(wire_vertex)
    wire_line = np.vstack(wire_line)
    vertex_vertex = np.vstack(vertex_vertex)
    mesh_model.export(out_folder / f"model.obj")
    with open(out_folder / f"wire.obj", "w") as f:
        for v in wire_vertex:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for l in wire_line:
            f.write(f"l {l[0]} {l[1]}\n")
    trimesh.PointCloud(vertex_vertex).export(out_folder / f"vertex.obj")
    pass