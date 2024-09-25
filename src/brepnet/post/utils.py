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

from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_Cylinder, GeomAbs_Plane
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
from OCC.Core.TColgp import TColgp_SequenceOfXY, TColgp_SequenceOfXYZ
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Core.Geom import Geom_Line
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from OCC.Core.Geom import Geom_BSplineCurve

# FIX_TOLERANCE = 1e-6
# CONNECT_TOLERANCE = 1e-3
# SEWING_TOLERANCE = 1e-3

# EDGE_FITTING_TOLERANCE = [5e-3, 8e-3, 5e-2]
# FACE_FITTING_TOLERANCE = [5e-2, 8e-2, 10e-2]

EDGE_FITTING_TOLERANCE = [5e-3, 8e-3, 5e-2]
FACE_FITTING_TOLERANCE = [5e-2, 8e-2, 8e-2]

FIX_TOLERANCE = 1e-2
FIX_PRECISION = 1e-2
CONNECT_TOLERANCE = 3e-2
SEWING_TOLERANCE = 3e-2
TRANSFER_PRECISION = 1e-3
MAX_DISTANCE_THRESHOLD = 1e-1
USE_VARIATIONAL_SMOOTHING = True
weight_CurveLength, weight_Curvature, weight_Torsion = 0.4, 0.4, 0.2
IS_VIZ_WIRE, IS_VIZ_FACE, IS_VIZ_SHELL = False, False, True
CONTINUITY = GeomAbs_C2


def get_edge_vertexes(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertexes.append(vertex)
        vertex_explorer.Next()
    return vertexes


def explore_edges(shape):
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edges.append(edge)
        edge_explorer.Next()
    return edges


def check_edges_similarity(edge1, edge2, dis_threshold=1e-1):
    def sample_edge(edge, sample_num=16):
        curve = BRepAdaptor_Curve(edge).Curve()
        first, last = curve.FirstParameter(), curve.LastParameter()
        points = []
        for i in range(sample_num):
            param = first + (last - first) * i / (sample_num - 1)
            point = curve.Value(param)
            points.append(point)
        return points

    edge1_sample_points = sample_edge(edge1)
    edge2_sample_points = sample_edge(edge2)

    if edge1_sample_points is None or edge2_sample_points is None:
        return False, -1

    dis_list1 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points):
        dis = p1.Distance(p2)
        dis_list1.append(dis)

    dis_list2 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points[::-1]):
        dis = p1.Distance(p2)
        dis_list2.append(dis)

    dis_list = dis_list1 if sum(dis_list1) < sum(dis_list2) else dis_list2

    is_similar = all([dis < dis_threshold for dis in dis_list])
    return is_similar, np.mean(dis_list)


def calculate_wire_bounding_box_length(wire):
    bbox = Bnd_Box()
    brepbnd = brepbndlib()
    brepbndlib.Add(wire, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = (xmax - xmin) + (ymax - ymin) + (zmax - zmin)
    return length


def viz_shapes(shapes, transparency=0):
    display, start_display, add_menu, add_function_to_menu = init_display()
    if len(shapes) == 1:
        display.DisplayShape(shapes[0], update=True, transparency=transparency)
    else:
        for shape in shapes:
            display.DisplayShape(shape, update=True, color=Colors.random_color(), transparency=transparency)
    display.FitAll()
    start_display()


def check_edges_in_face(face, edges, dist_tol=CONNECT_TOLERANCE):
    for edge in edges:
        edge_vertexes = get_edge_vertexes(edge)
        for vertex in edge_vertexes:
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, face)
            min_dist = dist_shape_shape.Value()
            if min_dist > dist_tol:
                return False
    return True


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    @staticmethod
    def random_color():
        return Quantity_Color(randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              Quantity_TOC_RGB)


def create_surface(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
    def fit_face(uv_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        deg_min, deg_max = 3, 8
        if use_variational_smoothing:
            # weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSplineSurface(uv_points_array, weight_CurveLength, weight_Curvature, weight_Torsion, deg_max,
                                                  CONTINUITY, precision).Surface()
        else:
            return GeomAPI_PointsToBSplineSurface(uv_points_array, deg_min, deg_max, CONTINUITY, precision).Surface()

    def set_face_uv_periodic(geom_face, points, tol=2e-3):
        u_intervals = np.sqrt(np.sum((points - np.roll(points, axis=0, shift=1)) ** 2, axis=2)).mean(axis=1)
        v_intervals = np.sqrt(np.sum((points - np.roll(points, axis=1, shift=1)) ** 2, axis=2)).mean(axis=0)

        u_min, u_max, v_min, v_max = geom_face.Bounds()
        us = geom_face.Value(u_min, v_min)
        ue = geom_face.Value(u_max, v_min)
        if us.Distance(ue) < np.mean(u_intervals):
            geom_face.SetUPeriodic()

        vs = geom_face.Value(u_min, v_min)
        ve = geom_face.Value(u_min, v_max)
        if vs.Distance(ve) < np.mean(v_intervals):
            geom_face.SetVPeriodic()
        return geom_face

    num_u_points, num_v_points = points.shape[0], points.shape[1]
    uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
    for u_index in range(1, num_u_points + 1):
        for v_index in range(1, num_v_points + 1):
            pt = points[u_index - 1, v_index - 1]
            point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            uv_points_array.SetValue(u_index, v_index, point_3d)

    precision = FACE_FITTING_TOLERANCE
    try:
        approx_face = fit_face(uv_points_array, precision[0], use_variational_smoothing)
    except Exception as e:
        try:
            approx_face = fit_face(uv_points_array, precision[1], use_variational_smoothing)
        except Exception as e:
            try:
                approx_face = fit_face(uv_points_array, precision[2], use_variational_smoothing)
            except Exception as e:
                approx_face = fit_face(uv_points_array, precision[-1], use_variational_smoothing)
    approx_face = set_face_uv_periodic(approx_face, points)

    return approx_face


def create_edge(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
    def fit_edge(u_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        deg_min, deg_max = 0, 8
        if use_variational_smoothing:
            # weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSpline(u_points_array, weight_CurveLength, weight_Curvature, weight_Torsion, deg_max,
                                           CONTINUITY, precision).Curve()
        else:
            return GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, CONTINUITY, precision).Curve()

    num_u_points = points.shape[0]
    u_points_array = TColgp_Array1OfPnt(1, num_u_points)
    for u_index in range(1, num_u_points + 1):
        pt = points[u_index - 1]
        point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        u_points_array.SetValue(u_index, point_2d)

    precision = EDGE_FITTING_TOLERANCE
    try:
        approx_edge = fit_edge(u_points_array, precision[0], use_variational_smoothing)
    except Exception as e:
        try:
            approx_edge = fit_edge(u_points_array, precision[1], use_variational_smoothing)
        except Exception as e:
            try:
                approx_edge = fit_edge(u_points_array, precision[2], use_variational_smoothing)
            except Exception as e:
                approx_edge = fit_edge(u_points_array, precision[-1], use_variational_smoothing)
    return approx_edge


def create_wire_from_unordered_edges(face_edges, max_retry_times=3, is_sort_by_length=True):
    wire_array = None
    for _ in range(max_retry_times):
        random.shuffle(face_edges)

        edges_seq = TopTools_HSequenceOfShape()
        for edge in face_edges:
            edges_seq.Append(edge)
        wire_array_c = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, CONNECT_TOLERANCE, False)

        # Check if all wires is valid
        all_wire_valid = True
        for i in range(1, wire_array_c.Length() + 1):
            wire_c = wire_array_c.Value(i)
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

    if wire_array is None or wire_array.Length() == 0:
        return None

    wire_list = [wire_array.Value(i) for i in range(1, wire_array.Length() + 1)]

    if is_sort_by_length:
        wire_list = sorted(wire_list, key=calculate_wire_bounding_box_length, reverse=True)

    return wire_list


def create_trimmed_face_from_wire(geom_face, wire_list):
    is_periodic = geom_face.IsUPeriodic() or geom_face.IsVPeriodic()

    face_fixer = ShapeFix_Face()
    face_fixer.Init(geom_face, CONNECT_TOLERANCE, True)
    wire_seq = TopTools_HSequenceOfShape()
    for wire in wire_list:
        wire_seq.Append(wire)
        face_fixer.Add(wire)

    face_fixer.FixWireTool().SetModifyGeometryMode(True)
    face_fixer.FixWireTool().SetMaxTolerance(CONNECT_TOLERANCE)
    face_fixer.FixWireTool().SetPrecision(FIX_PRECISION)
    face_fixer.FixWireTool().SetFixShiftedMode(True)
    face_fixer.FixWireTool().SetFixGaps2dMode(True)
    face_fixer.FixWireTool().SetFixGaps3dMode(True)
    face_fixer.FixWireTool().SetClosedWireMode(True)
    face_fixer.FixWireTool().SetFixTailMode(True)
    face_fixer.FixWireTool().Perform()
    face_fixer.FixWireTool().FixGaps2d()
    face_fixer.FixWireTool().FixGaps3d()
    face_fixer.FixWireTool().FixConnected()

    face_fixer.SetAutoCorrectPrecisionMode(False)
    face_fixer.SetPrecision(FIX_PRECISION)
    face_fixer.SetMaxTolerance(CONNECT_TOLERANCE)
    face_fixer.SetFixOrientationMode(True)
    face_fixer.SetFixMissingSeamMode(True)
    face_fixer.SetFixSplitFaceMode(True)
    face_fixer.SetFixWireMode(True)
    face_fixer.SetFixLoopWiresMode(True)
    face_fixer.SetFixIntersectingWiresMode(True)
    face_fixer.SetFixPeriodicDegeneratedMode(True)
    face_fixer.SetFixSmallAreaWireMode(True)
    face_fixer.SetRemoveSmallAreaFaceMode(True)
    face_fixer.Perform()

    try:
        face_fixer.FixAddNaturalBound()
        face_fixer.FixOrientation()
        face_fixer.FixMissingSeam()
        face_fixer.FixWiresTwoCoincEdges()
        face_fixer.FixIntersectingWires()
        face_fixer.FixPeriodicDegenerated()
        face_fixer.FixOrientation()
    except Exception as e:
        print(f"Error fixing face {e}")

    face_occ = face_fixer.Face()

    return face_occ


def create_trimmed_face1(geom_face, face_edges):
    wire_list = create_wire_from_unordered_edges(face_edges)
    if wire_list is None:
        return None, None, False
    trimmed_face = create_trimmed_face_from_wire(geom_face, wire_list)
    face_analyzer = BRepCheck_Analyzer(trimmed_face)
    is_face_valid = face_analyzer.IsValid()
    return wire_list, trimmed_face, is_face_valid


def create_trimmed_face2(geom_face, topo_face, face_edges):
    # try to find the replaced edge
    topo_face_edges = explore_edges(topo_face)
    replace_dict = {}
    for idx1, checked_edge in enumerate(face_edges):
        optional_edges = {}
        for idx2, replaced_edge in enumerate(topo_face_edges):
            is_similar, mean_dis = check_edges_similarity(checked_edge, replaced_edge)
            if is_similar:
                optional_edges[idx2] = mean_dis
        if len(optional_edges) == 0:
            continue
        optimal_edge_idx = min(optional_edges, key=optional_edges.get)
        replace_dict[idx1] = optimal_edge_idx
        face_edges[idx1] = topo_face_edges[optimal_edge_idx]
    return create_trimmed_face1(geom_face, face_edges)


def try_create_trimmed_face(geom_face, topo_face, face_edges):
    wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges)
    if is_face_valid1:
        return wire_list1, trimmed_face1, 0

    wire_list2, trimmed_face2, is_face_valid2 = create_trimmed_face2(geom_face, topo_face, face_edges)
    if is_face_valid2:
        return wire_list2, trimmed_face2, 1

    return wire_list1, topo_face, 2


# Fit parametric surfaces / curves and trim into B-rep
def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, folder_path, isdebug=False, is_save_face=True, debug_face_idx=[]):
    print(f"{Colors.GREEN}################################ 1. Fit primitives ################################{Colors.RESET}")
    recon_geom_faces = [create_surface(points) for points in surf_wcs]
    recon_topo_faces = [BRepBuilderAPI_MakeFace(geom_face, 1e-3).Face() for geom_face in recon_geom_faces]
    recon_curves = [create_edge(points) for points in edge_wcs]
    recon_edge = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in recon_curves]

    # if isdebug:
    #     viz_shapes(recon_geom_faces, transparency=0.5)
    #     os.makedirs(os.path.join(folder_path, 'fitting_geom_face'), exist_ok=True)
    #     for idx, geom_face in enumerate(recon_geom_faces):
    #         if idx != 15:
    #             continue
    #         topo_face_from_geom = BRepBuilderAPI_MakeFace(geom_face, 1e-3).Face()
    #         # write_stl_file(topo_face_from_geom, os.path.join(folder_path, 'fitting_face', f'{idx}.stl'), linear_deflection=0.1,
    #         #                angular_deflection=0.5)
    #         viz_shapes([topo_face_from_geom], transparency=0.5)

    print(f"{Colors.GREEN}################################ 2. Trim Face ######################################{Colors.RESET}")
    # Cut surface by wire
    is_face_success_list = []
    trimmed_faces = []
    for idx, (geom_face, topo_face, face_edge_idx) in enumerate(zip(recon_geom_faces, recon_topo_faces, FaceEdgeAdj)):
        face_edges = [recon_edge[edge_idx] for edge_idx in face_edge_idx]
        if idx in debug_face_idx:
            is_viz_wire, is_viz_face, is_viz_shell = True, True, True
        else:
            is_viz_wire, is_viz_face, is_viz_shell = IS_VIZ_WIRE, IS_VIZ_FACE, IS_VIZ_SHELL
        if not isdebug:
            is_viz_wire, is_viz_face, is_viz_shell = False, False, False

        # 3. Construct face using geom surface and wires
        wire_list, trimmed_face, method_type = try_create_trimmed_face(geom_face, topo_face, face_edges)

        # visualize the constructed wire
        if is_viz_wire:
            shapes = wire_list
            display, start_display, add_menu, add_function_to_menu = init_display()
            if len(shapes) == 1:
                display.DisplayShape(shapes[0], update=True)
            else:
                for shape in shapes:
                    display.DisplayShape(shape, update=True, color=Colors.random_color())
            display.FitAll()
            start_display()
            # viz_shapes(wire_list)

        if is_viz_face:
            shapes = [trimmed_face]
            display, start_display, add_menu, add_function_to_menu = init_display()
            if len(shapes) == 1:
                display.DisplayShape(shapes[0], update=True)
            else:
                for shape in shapes:
                    display.DisplayShape(shape, update=True, color=Colors.random_color())
            display.FitAll()
            start_display()
            # viz_shapes([trimmed_face])

        # check if the face contains all edges
        is_face_success = check_edges_in_face(trimmed_face, face_edges)
        is_face_success_list.append(is_face_success)
        if is_face_success:
            trimmed_faces.append(trimmed_face)
        else:
            trimmed_faces.append(trimmed_face)

        if isdebug and not is_face_success:
            print(f"{Colors.RED}Folder_path: {folder_path}, Face {idx} is not valid{Colors.RESET}")
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(trimmed_face, update=True)
            for edge in face_edges:
                display.DisplayShape(edge, update=True)
            display.FitAll()
            start_display()

        # save the face as step file and stl file
        if is_save_face and is_face_success:
            os.makedirs(os.path.join(folder_path, 'recon_face'), exist_ok=True)
            try:
                write_step_file(trimmed_face, os.path.join(folder_path, 'recon_face', f'{idx}.step'))
                write_stl_file(trimmed_face, os.path.join(folder_path, 'recon_face', f'{idx}.stl'), linear_deflection=0.1,
                               angular_deflection=0.5)
            except:
                print(f"Error writing step or stl file for face {idx}")

    print(f"{Colors.GREEN}################################ 3. Sew solid ################################{Colors.RESET}")
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(SEWING_TOLERANCE)
    for face in trimmed_faces:
        sewing.Add(face)
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    if isdebug and is_viz_shell:  # sewn_shell.ShapeType() == TopAbs_COMPOUND:
        viz_shapes([sewn_shell])

    if sewn_shell.ShapeType() == TopAbs_COMPOUND:
        return sewn_shell, is_face_success_list

    # fix the shell
    fix_shell = ShapeFix_Shell(sewn_shell)
    fix_shell.SetPrecision(FIX_PRECISION)
    fix_shell.SetMaxTolerance(FIX_TOLERANCE)
    fix_shell.SetFixFaceMode(True)
    fix_shell.SetFixOrientationMode(True)
    fix_shell.Perform()
    sewn_shell = fix_shell.Shell()

    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    fix_solid = ShapeFix_Solid(solid)
    fix_solid.SetPrecision(FIX_TOLERANCE)
    fix_solid.SetMaxTolerance(FIX_TOLERANCE)
    fix_solid.SetFixShellMode(True)
    fix_solid.SetFixShellOrientationMode(True)
    fix_solid.SetCreateOpenSolidMode(False)
    fix_solid.Perform()
    fixed_solid = fix_solid.Solid()

    print(f"{Colors.GREEN}################################ Construct Done ################################{Colors.RESET}")
    return fixed_solid, is_face_success_list
