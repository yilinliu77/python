import os
import shutil

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

def import_step_file_as_obj(file_path, output_folder, step_line_deflection, step_angle_deflection, step_edge_sample):
    step_reader = STEPControl.STEPControl_Reader()
    
    status = step_reader.ReadFile(str(file_path))
    if status != 1:
        raise Exception("Error: Cannot read the STEP file.")
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    # if shape.ShapeType() in [TopAbs.TopAbs_COMPOUND, TopAbs.TopAbs_COMPSOLID]:
        # raise Exception("Error: Wrong Shape Type, cannot support COMPOUND or COMPSOLID.")

    # Model
    try:
        mesh = BRepMesh_IncrementalMesh(shape, 2e-1, False, 2e-1)
    except:
        print("Precision Error")
        return
    mesh.Perform()

    # Convert into a temporary STL file first
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as stl_file:
        stl_writer = StlAPI_Writer()
        stl_writer.Write(shape, stl_file.name)
        stl_file_path = stl_file.name
    
    # Then convert into obj file
    mesh = trimesh.load_mesh(stl_file_path)
    
    vertices = np.asarray(mesh.vertices)
    bbox = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    scale = np.max(bbox[1] - bbox[0])
    center = np.mean(bbox, axis=0)
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale)
    
    mesh.fix_normals()  # Recalculates normals
    
    # root, _ = os.path.splitext(file_path)
    file_name = file_path.stem
    # file_name = os.path.basename(root)
    
    temp_dir = os.path.join(output_folder, file_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    model_file_path = os.path.join(temp_dir, file_name + "_model.obj")
    obj_output = trimesh.exchange.obj.export_obj(mesh, mtl_name=f"{file_name}_model")
    with open(model_file_path, 'w') as model_file:
        model_file.write(obj_output)
        
    # Clean up temporary STL file
    os.remove(stl_file_path)
    
    # Wire
    edges = []
    exp = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_EDGE)
    while(exp.More()):
        edge = TopoDS.topods.Edge(exp.Current())
        edges.append(edge)
        exp.Next()
    wire_file_path = os.path.join(temp_dir, file_name + "_wire.obj")
    with open(wire_file_path, 'w') as wire_file:
        wire_file.write(f"# Wire data, {file_name} \n")
        vertex_index = 1
        for edge in edges:
            curve_adaptor = BRepAdaptor_Curve(edge)
            first = curve_adaptor.FirstParameter()
            last = curve_adaptor.LastParameter()
            # Sample points along the edge
            for i in range(step_edge_sample):
                u = first + (last - first) * (i / (step_edge_sample - 1))  # Parameter for sampling
                p = gp_Pnt()
                curve_adaptor.D0(u, p)

                p = np.asarray([p.X(), p.Y(), p.Z()])
                p = (p - center) / scale

                # Write the vertex to the OBJ file
                wire_file.write(f"v {p[0]} {p[1]} {p[2]}\n")

            # Write the lines connecting the sampled points
            for i in range(vertex_index, vertex_index + step_edge_sample - 1):
                wire_file.write(f"l {i} {i + 1}\n")

            vertex_index += step_edge_sample
    
    # Vertex
    vertices = []
    exp = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_VERTEX)
    while exp.More():
        vertex = TopoDS.topods.Vertex(exp.Current())
        vertices.append(vertex)
        exp.Next()
    
    vertex_file_path = os.path.join(temp_dir, file_name + "_vertex.obj")
    with open(vertex_file_path, 'w') as vertex_file:
        vertex_file.write(f"# Vertex data, {file_name} \n")
        for vertex in vertices:
            p = BRep_Tool.Pnt(vertex)
            p = np.asarray([p.X(), p.Y(), p.Z()])
            p = (p - center) / scale
            vertex_file.write(f"v {p[0]} {p[1]} {p[2]}\n")
    return 

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step_to_obj.py <step_folder> <out_folder>")
        sys.exit(1)
    step_folder = Path(sys.argv[1])
    out_folder = Path(sys.argv[2])
    os.makedirs(out_folder, exist_ok=True)
    for step_file in tqdm(step_folder.rglob("*.step")):
        # folder_name = step_file.parent.name
        # shutil.copyfile(step_file, out_folder / f"{folder_name}.step")
        if step_file.name.endswith(".step"):
            import_step_file_as_obj(step_file, out_folder, 0.01, 0.005, 100)
