import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods, topods_Vertex
from OCC.Display.SimpleGui import init_display
from OCC.Core.GeomConvert import geomconvert

from OCC.Core.GeomAbs import (GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface)


from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.DataExchange import read_step_file

import numpy as np

if __name__ == '__main__':
    display, start_display, add_menu, add_function_to_menu = init_display()
    step_filename = 'G:/Dataset/ABC/raw_data/abc_0000_obj_v00/00000007/00000007_b33a147f86da49879455d286_step_000.step'
    shape = read_step_file(step_filename)

    # Function to explore and print the elements of a shape
    def explore_shape(shape, shape_type):
        explorer = TopExp_Explorer(shape, shape_type)
        while explorer.More():
            yield explorer.Current()
            explorer.Next()

    # Explore and list faces, edges, and vertices
    faces = list(explore_shape(shape, TopAbs_FACE))
    edges = list(explore_shape(shape, TopAbs_EDGE))
    vertices = list(explore_shape(shape, TopAbs_VERTEX))

    # Sample points in face
    face_dict = {}
    face_sample_points = []
    for face in explore_shape(shape, TopAbs_FACE):
        face_dict[face] = len(face_dict)
        surface = BRepAdaptor_Surface(face)

        if surface.GetType() not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface]:
            continue

        first_u = surface.FirstUParameter()
        last_u = surface.LastUParameter()
        first_v = surface.FirstVParameter()
        last_v = surface.LastVParameter()
        u = np.linspace(first_u, last_u, num=20)
        v = np.linspace(first_v, last_v, num=20)
        u, v = np.meshgrid(u, v)
        points = []
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                pnt = surface.Value(u[i, j], v[i, j])
                points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
        face_sample_points.append(np.stack(points, axis=0).reshape(20,20,3))
    face_sample_points = np.stack(face_sample_points, axis=0)

    assert len(face_dict) == len(faces)

    edge_dict = {}
    edge_sample_points = []
    # Sample points in face
    for edge in explore_shape(shape, TopAbs_EDGE):
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve]:
            continue
        edge_dict[edge] = len(edge_dict)
        # Sample 20 points along it
        range_start = curve.FirstParameter()
        range_end = curve.LastParameter()
        sample_u = np.linspace(range_start, range_end, num=20)
        sample_points=[]
        for u in sample_u:
            pnt = curve.Value(u)
            sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
        edge_sample_points.append(np.stack(sample_points, axis=0))
    edge_sample_points = np.stack(edge_sample_points, axis=0)

    assert len(edge_dict) == len(edges)

    # Connectivity
    edge_connectivity = []
    edge_vertex_map = {}
    for edge in explore_shape(shape, TopAbs_EDGE):
        if edge not in edge_dict:
            continue
        # Get vertices of the current edge
        for vertex in explore_shape(edge, TopAbs_VERTEX):
            if vertex not in edge_vertex_map:
                edge_vertex_map[vertex] = [edge]
            else:
                edge_vertex_map[vertex].append(edge)
    for vertex, edges in edge_vertex_map.items():
        for i in range(len(edges)):
            id1 = edge_dict[edges[i]]
            id2 = edge_dict[edges[(i+1)%len(edges)]]
            if id1 < id2:
                edge_connectivity.append([id1, id2])
            else:
                edge_connectivity.append([id2, id1])

    face_connectivity = np.zeros((len(faces), len(faces)), dtype=np.int8)
    face_edge_map = {}
    for face in explore_shape(shape, TopAbs_FACE):
        if face not in face_dict:
            continue
        # Get edges of the current face
        for edge in explore_shape(face, TopAbs_EDGE):
            if edge not in face_edge_map:
                face_edge_map[edge] = [face]
            else:
                face_edge_map[edge].append(face)
    for edge, faces in face_edge_map.items():
        for i in range(len(faces)):
            id1 = face_dict[faces[i]]
            id2 = face_dict[faces[(i+1)%len(faces)]]
            face_connectivity[id1, id2] = 1
            face_connectivity[id2, id1] = 1

    face_edge_connectivity = []
    for face in explore_shape(shape, TopAbs_FACE):
        loops = []

        for wire in explore_shape(face, TopAbs_WIRE):
            loops.append(-2)
            for edge in explore_shape(wire, TopAbs_EDGE):
                if edge not in edge_dict:
                    continue
                loops.append(edge_dict[edge])
        face_edge_connectivity.append(loops)

    display.DisplayShape(shape, update=True)
    display.FitAll()
    start_display()