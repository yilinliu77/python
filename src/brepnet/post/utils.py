import argparse
import copy
import math
import os
import queue
import random
import string
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing

from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge, ShapeFix_Shell, ShapeFix_Solid, \
    ShapeFix_ComposeShell
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.gp import gp_Pnt, gp_XYZ, gp_Vec
from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCC.Extend.DataExchange import write_stl_file, write_step_file

# xdt
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from OCC.Core.ShapeAnalysis import ShapeAnalysis_WireOrder
from OCC.Core.BRepAlgo import BRepAlgo_Loop

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from random import randint
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepCheck import BRepCheck_Analyzer

from OCC.Core.GeomPlate import (GeomPlate_BuildPlateSurface, GeomPlate_PointConstraint, GeomPlate_CurveConstraint,
                                GeomPlate_MakeApprox, GeomPlate_PlateG0Criterion, GeomPlate_PlateG1Criterion, )
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TColgp import TColgp_SequenceOfXY, TColgp_SequenceOfXYZ
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Core.Geom import Geom_Line
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from OCC.Core.Geom import Geom_BSplineCurve

FIX_TOLERANCE = 1e-6
CONNECT_TOLERANCE = 1e-3
SEWING_TOLERANCE = 1e-3


# FIX_TOLERANCE = 1e-7
# CONNECT_TOLERANCE = 1e-7
# SEWING_TOLERANCE = 1e-7


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face, ignore_orientation=True)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, FIX_TOLERANCE)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, FIX_TOLERANCE)

        ok = wire_fixer.Perform()
        # assert ok
        # if not ok:
        #     display_trim_faces([face])


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(FIX_TOLERANCE)
    fixer.SetMaxTolerance(FIX_TOLERANCE)
    # fixer.SetAutoCorrectPrecisionMode(True)
    # fixer.SetFixMissingSeamMode(True)
    # fixer.SetFixOrientationMode(True)
    # fixer.SetFixIntersectingWiresMode(True)
    # fixer.SetFixLoopWiresMode(True)
    # fixer.FixAddNaturalBound()
    ok = fixer.Perform()
    fixer.FixMissingSeam()
    fixer.FixAddNaturalBound()
    fixer.FixOrientation()
    fixer.FixIntersectingWires()
    fixer.FixPeriodicDegenerated()
    face = fixer.Face()
    return face


def display_trim_faces(post_faces):
    # Sew faces into solid
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)

    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # display it
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(sewn_shell, update=True)
    display.FitAll()
    start_display()


def display_edges(topods_edges):
    display, start_display, add_menu, add_function_to_menu = init_display()
    for edge in topods_edges:
        display.DisplayShape(edge, update=True)
    display.FitAll()
    start_display()


def get_edge_points(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    points = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        point = BRep_Tool.Pnt(vertex)
        points.append((point.X(), point.Y(), point.Z()))
        vertex_explorer.Next()
    return points


def get_edge_vertexes(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertexes.append(vertex)
        vertex_explorer.Next()
    return vertexes


def calculate_wire_bounding_box_length(wire):
    bbox = Bnd_Box()
    brepbnd = brepbndlib()
    brepbndlib.Add(wire, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = (xmax - xmin) + (ymax - ymin) + (zmax - zmin)
    return length


def points_are_close(p1, p2, tolerance=1e-7):
    return (abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance and abs(p1[2] - p2[2]) < tolerance)


def random_color():
    return Quantity_Color(randint(0, 255) / 255.0,
                          randint(0, 255) / 255.0,
                          randint(0, 255) / 255.0,
                          Quantity_TOC_RGB)


def points_are_equal(pnt1, pnt2, tolerance=1e-6):
    return pnt1.Distance(pnt2) < tolerance


def are_edges_almost_coincident(edge1, edge2, ang_tolerance=2 * 3.14 / 180, dis_tolerance=1e-3):
    vertex1_1, vertex1_2 = topexp.FirstVertex(edge1), topexp.LastVertex(edge1)
    vertex2_1, vertex2_2 = topexp.FirstVertex(edge2), topexp.LastVertex(edge2)

    pnt1_1, pnt1_2 = BRep_Tool.Pnt(vertex1_1), BRep_Tool.Pnt(vertex1_2)
    pnt2_1, pnt2_2 = BRep_Tool.Pnt(vertex2_1), BRep_Tool.Pnt(vertex2_2)

    vec1 = gp_Vec(pnt1_1, pnt1_2)
    vec2 = gp_Vec(pnt2_1, pnt2_2)

    return (vec1.IsParallel(vec2.Reversed(), ang_tolerance) and
            points_are_equal(pnt1_2, pnt2_1, dis_tolerance) and
            points_are_equal(pnt1_1, pnt2_2, dis_tolerance))


def has_opposite_coincident_edges(wire, tolerance=1e-3):
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    edges = []

    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()

    for i in range(len(edges)):
        next_index = (i + 1) % len(edges)  # 包括起点和终点
        if are_edges_almost_coincident(edges[i], edges[next_index], tolerance):
            return True

    return False


def has_self_intersections(wire):
    analysis = ShapeAnalysis_Wire()
    analysis.Load(wire)
    return analysis.CheckSelfIntersection()


def viz_shape(shape):
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    start_display()


def print_edge(edge_list, is_viz=False, reverse=False):
    print("Edge list:")
    for edge in edge_list:
        if reverse:
            edge = edge.Reversed()
        curve_handle, first, last = BRep_Tool.Curve(edge)
        point_start = curve_handle.Value(first)
        point_end = curve_handle.Value(last)
        print(f"Edge from {point_start.XYZ().Coord()} to {point_end.XYZ().Coord()}")
        if is_viz:
            viz_shape(edge)


def explore_edge(shape):
    edges = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        edges.append(edge)
        explorer.Next()
    return edges


def unify_edge_endpoints(edge_wcs, tol):
    points = edge_wcs[:, [0, -1]].reshape(-1, 3)  # N*2

    merged_points = []
    visited = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if visited[i]:
            continue
        distances = np.linalg.norm(points - point, axis=1)
        close_points = points[distances <= tol]
        merged_point = close_points.mean(axis=0)
        merged_points.append(merged_point)
        visited[distances <= tol] = True
    merged_points = np.array(merged_points)

    for i in range(len(edge_wcs)):
        edge_wcs[i][0] = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][0], axis=1))]
        edge_wcs[i][-1] = merged_points[np.argmin(np.linalg.norm(merged_points - edge_wcs[i][-1], axis=1))]

    return edge_wcs


def compute_mass(geom_face, wire):
    face_builder = BRepBuilderAPI_MakeFace(geom_face, wire)
    props = GProp_GProps()
    face_fix = ShapeFix_Face(face_builder.Face())
    face_fix.SetFixOrientationMode(False)
    face_fix.Perform()
    # viz_shape(face_builder.Face())
    brepgprop.SurfaceProperties(face_builder.Face(), props)
    area = props.Mass()
    return area


# single loop
def create_wire(face, edges):
    topo_wire = ShapeExtend_WireData()
    for edge in edges:
        topo_wire.Add(edge)

    fix_wire = ShapeFix_Wire()
    fix_wire.Load(topo_wire)
    fix_wire.SetSurface(face)

    fix_wire.FixReorder()
    fix_wire.FixConnected()
    fix_wire.FixClosed()
    fix_wire.FixEdgeCurves()
    fix_wire.FixDegenerated()
    fix_wire.FixSelfIntersection()
    fix_wire.FixLacking()
    fix_wire.FixNotchedEdges()
    fix_wire.FixConnected()

    fix_wire.Perform()
    # print(fix_wire.NbEdges())

    return fix_wire.WireAPIMake()


"""
Fit parametric surfaces / curves and trim into B-rep
"""


def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, folder_path, isdebug=False, is_save_face=True, debug_face_idx=[]):
    # Fit surface bspline
    recon_faces = []
    for points in surf_wcs:
        num_u_points, num_v_points = surf_wcs.shape[1], surf_wcs.shape[2]
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1, num_u_points + 1):
            for v_index in range(1, num_v_points + 1):
                pt = points[u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)

        precision = [1e-10, 1e-8, 1e-6, 1e-3, 1e-2]
        deg_min, deg_max = 3, 48
        try:
            approx_face = GeomAPI_PointsToBSplineSurface(uv_points_array, deg_min, deg_max, GeomAbs_C2, precision[0]).Surface()
        except Exception as e:
            try:
                approx_face = GeomAPI_PointsToBSplineSurface(uv_points_array, deg_min, deg_max, GeomAbs_C2, precision[1]).Surface()
            except Exception as e:
                try:
                    approx_face = GeomAPI_PointsToBSplineSurface(
                            uv_points_array, deg_min, deg_max, GeomAbs_C2, precision[2]).Surface()
                except Exception as e:
                    try:
                        approx_face = GeomAPI_PointsToBSplineSurface(
                                uv_points_array, deg_min, deg_max, GeomAbs_C2, precision[3]).Surface()
                    except Exception as e:
                        approx_face = GeomAPI_PointsToBSplineSurface(
                                uv_points_array, deg_min, deg_max, GeomAbs_C2, precision[-1]).Surface()
        if approx_face.IsUClosed():
            approx_face.SetUPeriodic()
        if approx_face.IsVClosed():
            approx_face.SetVPeriodic()
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = edge_wcs.shape[1]
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1, num_u_points + 1):
            pt = points[u_index - 1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)

        precision = [1e-10, 1e-8, 1e-6, 1e-3, 1e-2]
        # precision = [5e-3, 8e-3, 5e-2]
        deg_min, deg_max = 0, 48

        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision[0]).Curve()
        except Exception as e:
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision[1]).Curve()
            except Exception as e:
                try:
                    approx_edge = GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision[2]).Curve()
                except Exception as e:
                    try:
                        approx_edge = GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision[3]).Curve()
                    except Exception as e:
                        approx_edge = GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision[-1]).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire
    is_face_success_list = []
    post_faces = []
    for idx, (surface, edge_indices) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        face_edges = [edge_list[edge_idx] for edge_idx in edge_indices]
        if idx in debug_face_idx:
            is_viz_wire, is_viz_face, is_viz_shell = True, True, True
        else:
            is_viz_wire, is_viz_face, is_viz_shell = False, False, False

        # 2. Construct wires from edges
        retry_times = 10  # Might be helpful
        wire_array = None
        for _ in range(retry_times):
            random.shuffle(face_edges)

            # Create wires from the edge list
            edges_seq = TopTools_HSequenceOfShape()
            for edge in face_edges:
                edges_seq.Append(edge)
            wire_array_c = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, CONNECT_TOLERANCE, False)

            # Check if all wires is valid
            all_wire_valid = True
            for i in range(1, wire_array_c.Length() + 1):
                wire_c = wire_array_c.Value(i)
                # print_edge(explore_edge(wire_c), is_viz=False)
                wire_analyzer = BRepCheck_Analyzer(wire_c)
                if not wire_analyzer.IsValid():
                    all_wire_valid = False
                    break
            if not all_wire_valid:
                break

            if wire_array is None:
                wire_array = wire_array_c
                continue

            if wire_array_c.Length() < wire_array.Length():
                wire_array = wire_array_c

        # 3. visualize the constructed wire
        if is_viz_wire:
            display, start_display, add_menu, add_function_to_menu = init_display()
            for i in range(1, wire_array.Length() + 1):
                wire_c = wire_array.Value(i)
                display.DisplayShape(wire_c, update=True, color=random_color())
            display.FitAll()
            start_display()

        # 4. Sort the wires by bounding box length to distinguish the outer and inner wires
        wire_list = []
        box_length_list = []
        for i in range(1, wire_array.Length() + 1):
            wire_c = wire_array.Value(i)
            wire_list.append(wire_c)
            length_c = calculate_wire_bounding_box_length(wire_c)
            box_length_list.append(length_c)
        sorted_wire_list = [x for _, x in sorted(zip(box_length_list, wire_list), key=lambda pair: pair[0], reverse=True)]

        # 5. Construct face using geom surface and wires
        face_fixer = ShapeFix_Face()
        face_fixer.Init(surface, CONNECT_TOLERANCE, True)
        for wire in sorted_wire_list:
            face_fixer.Add(wire)
        face_fixer.SetAutoCorrectPrecisionMode(True)
        face_fixer.SetFixOrientationMode(True)
        face_fixer.SetFixSplitFaceMode(True)
        face_fixer.SetFixMissingSeamMode(True)
        face_fixer.SetFixWireMode(True)
        face_fixer.SetFixLoopWiresMode(True)
        face_fixer.SetFixIntersectingWiresMode(True)
        face_fixer.SetFixPeriodicDegeneratedMode(True)
        face_fixer.Perform()
        face_fixer.FixMissingSeam()
        face_fixer.FixAddNaturalBound()
        face_fixer.FixOrientation()
        face_fixer.FixIntersectingWires()
        face_fixer.FixPeriodicDegenerated()
        face_occ = face_fixer.Face()
        post_faces.append(face_occ)

        # display_trim_faces([face_occ])
        if is_viz_face:
            # write_stl_file(face_occ, 'debug.stl', linear_deflection=0.001, angular_deflection=0.5)
            _post_faces = [face_occ]
            # Sew faces into solid
            sewing = BRepBuilderAPI_Sewing()
            # sewing.SetTolerance(1e-3)
            for face in _post_faces:
                sewing.Add(face)

            # Perform the sewing operation
            sewing.Perform()
            sewn_shell = sewing.SewedShape()

            # display it
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(sewn_shell, update=True)
            display.FitAll()
            start_display()

        # check if the face contains all edges
        is_face_success = True
        for edge_idx in edge_indices:
            # check if all the control vertexes of each edge are in the face
            is_edge_success = True
            points = get_edge_vertexes(edge_list[edge_idx])
            for point in points:
                dist_shape_shape = BRepExtrema_DistShapeShape(point, face_occ)
                min_dist = dist_shape_shape.Value()
                if min_dist > 1e-3:
                    is_edge_success = False
                    break
            if is_edge_success == False:
                is_face_success = False
                break

        if isdebug and not is_face_success:
            print(f"folder_path: {folder_path}, Face {idx} is not valid")
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(face_occ, update=True)
            for edge in face_edges:
                display.DisplayShape(edge, update=True)
            display.FitAll()
            start_display()

        # save the face as step file and stl file
        if is_save_face and is_face_success:
            os.makedirs(os.path.join(folder_path, 'recon_face'), exist_ok=True)
            try:
                write_step_file(face_occ, os.path.join(folder_path, 'recon_face', f'{idx}.step'))
                write_stl_file(face_occ, os.path.join(folder_path, 'recon_face', f'{idx}.stl'), linear_deflection=0.001,
                               angular_deflection=0.5)
            except:
                print(f"Error writing step or stl file for face {idx}")

        is_face_success_list.append(is_face_success)
        pass

    # Sew faces into solid
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(SEWING_TOLERANCE)
    for face in post_faces:
        sewing.Add(face)

    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    if isdebug and is_viz_shell:  # sewn_shell.ShapeType() == TopAbs_COMPOUND:
        # display it
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(sewn_shell, update=True)
        display.FitAll()
        start_display()

    if sewn_shell.ShapeType() == TopAbs_COMPOUND:
        return sewn_shell, is_face_success_list

    # fix the shell
    # fix_shell = ShapeFix_Shell(sewn_shell)
    # fix_shell.SetPrecision(FIX_TOLERANCE)
    # fix_shell.SetMaxTolerance(FIX_TOLERANCE)
    # fix_shell.SetFixFaceMode(True)
    # fix_shell.SetFixOrientationMode(True)
    # fix_shell.Perform()
    # sewn_shell = fix_shell.Shell()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    # fix the soild
    fix_solid = ShapeFix_Solid(solid)
    fix_solid.SetPrecision(FIX_TOLERANCE)
    fix_solid.SetMaxTolerance(FIX_TOLERANCE)
    fix_solid.SetFixShellMode(True)
    fix_solid.SetFixShellOrientationMode(True)
    fix_solid.SetCreateOpenSolidMode(True)
    fix_solid.Perform()
    fixed_solid = fix_solid.Solid()

    return fixed_solid, is_face_success_list
