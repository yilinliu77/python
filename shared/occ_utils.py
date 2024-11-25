import numpy as np
from OCC.Core import BRepBndLib, TopoDS
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Vec, gp_Trsf


def disable_occ_log():
    from OCC.Core.Message import Message_Alarm, message
    printers = message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)


def normalize_shape(v_shape, v_bounding=1.):
    boundingBox = Bnd_Box()
    BRepBndLib.brepbndlib.Add(v_shape, boundingBox)
    xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()
    scale_x = v_bounding * 2 / (xmax - xmin)
    scale_y = v_bounding * 2 / (ymax - ymin)
    scale_z = v_bounding * 2 / (zmax - zmin)
    scaleFactor = min(scale_x, scale_y, scale_z)
    translation1 = gp_Vec(-(xmax + xmin) / 2, -(ymax + ymin) / 2, -(zmax + zmin) / 2)
    trsf1 = gp_Trsf()
    trsf1.SetTranslationPart(translation1)
    trsf2 = gp_Trsf()
    trsf2.SetScaleFactor(scaleFactor)
    trsf2.Multiply(trsf1)

    transformer = BRepBuilderAPI_Transform(trsf2)
    transformer.Perform(v_shape)
    shape = transformer.Shape()
    return shape


def get_triangulations(v_shape, v_resolution1=0.1, v_resolution2=0.1):
    if v_resolution1 > 0:
        mesh = BRepMesh_IncrementalMesh(v_shape, v_resolution1, False, v_resolution2)
    v = []
    f = []
    face_explorer = TopExp_Explorer(v_shape, TopAbs_FACE)
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
                f.append([t.Value(3) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                          t.Value(1) + cur_vertex_size - 1])
            else:
                f.append([t.Value(1) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                          t.Value(3) + cur_vertex_size - 1])
        face_explorer.Next()
    return v, f


def get_curve_length(edge, num=100):
    curve = BRepAdaptor_Curve(edge)
    range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
    range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()

    sample_u = np.linspace(range_start, range_end, num=num)
    last_point = None
    length = 0
    for u in sample_u:
        pnt = curve.Value(u)
        if last_point is not None:
            length += np.linalg.norm(np.array([pnt.X(), pnt.Y(), pnt.Z()]) - last_point)
            last_point = np.array([pnt.X(), pnt.Y(), pnt.Z()])
        else:
            last_point = np.array([pnt.X(), pnt.Y(), pnt.Z()])
    return length


def get_points_along_edge(edge, v_num=100):
    curve = BRepAdaptor_Curve(edge)
    range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
    range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()

    sample_u = np.linspace(range_start, range_end, num=v_num)
    points = []
    for u in sample_u:
        pnt = curve.Value(u)
        v1 = gp_Vec()
        curve.D1(u, pnt, v1)
        v1 = v1.Normalized()
        points.append(np.array([pnt.X(), pnt.Y(), pnt.Z(), v1.X(), v1.Y(), v1.Z()], dtype=np.float32))
    points = np.array(points, dtype=np.float32)
    return points



def get_primitives(v_shape, v_type):
    assert v_shape is not None
    explorer = TopExp_Explorer(v_shape, v_type)
    items = []
    while explorer.More():
        items.append(explorer.Current())
        explorer.Next()
    return items


def get_ordered_edges(v_face):
    edges = []
    for wire in get_primitives(v_face, TopAbs_WIRE):
        wire_explorer = BRepTools_WireExplorer(wire)
        local_edges = []
        while wire_explorer.More():
            edge = TopoDS.topods.Edge(wire_explorer.Current())
            local_edges.append(edge)
            wire_explorer.Next()
        edges.append(local_edges)
    return edges
