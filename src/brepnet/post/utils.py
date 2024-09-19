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

# FIX_TOLERANCE = 1e-6
# CONNECT_TOLERANCE = 1e-3
# SEWING_TOLERANCE = 1e-3

FITTING_TOLERANCE = 1e-5
FIX_TOLERANCE = 1e-3
CONNECT_TOLERANCE = 1e-3
SEWING_TOLERANCE = 1e-3
USE_VARIATIONAL_SMOOTHING = False


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


def viz_shapes(shapes, transparency=0):
    display, start_display, add_menu, add_function_to_menu = init_display()
    if len(shapes) == 1:
        display.DisplayShape(shapes[0], update=True, transparency=transparency)
    else:
        for shape in shapes:
            display.DisplayShape(shape, update=True, color=Colors.random_color(), transparency=transparency)
    display.FitAll()
    start_display()


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
    def fit_face(uv_points_array, precision, use_variational_smoothing=True):
        deg_min, deg_max = 3, 8
        if use_variational_smoothing:
            weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSplineSurface(uv_points_array, weight_CurveLength, weight_Curvature, weight_Torsion, deg_max,
                                                  GeomAbs_C2, precision).Surface()
        else:
            return GeomAPI_PointsToBSplineSurface(uv_points_array, deg_min, deg_max, GeomAbs_C2, precision).Surface()

    def set_face_uv_periodic(geom_face, tol=CONNECT_TOLERANCE):
        u_min, u_max, v_min, v_max = geom_face.Bounds()
        us = geom_face.Value(u_min, v_min)
        ue = geom_face.Value(u_max, v_min)
        if us.Distance(ue) < tol:
            geom_face.SetUPeriodic()
        vs = geom_face.Value(u_min, v_min)
        ve = geom_face.Value(u_min, v_max)
        if vs.Distance(ve) < tol:
            geom_face.SetVPeriodic()
        return geom_face

    num_u_points, num_v_points = points.shape[0], points.shape[1]
    uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
    for u_index in range(1, num_u_points + 1):
        for v_index in range(1, num_v_points + 1):
            pt = points[u_index - 1, v_index - 1]
            point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            uv_points_array.SetValue(u_index, v_index, point_3d)

    precision = [FITTING_TOLERANCE, FITTING_TOLERANCE * 2, FITTING_TOLERANCE * 5, FITTING_TOLERANCE * 10]
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
    approx_face = set_face_uv_periodic(approx_face)

    return approx_face


def create_edge(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
    def fit_edge(u_points_array, precision, use_variational_smoothing=True):
        deg_min, deg_max = 0, 8
        if use_variational_smoothing:
            weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSpline(u_points_array, weight_CurveLength, weight_Curvature, weight_Torsion, deg_max,
                                           GeomAbs_C2, precision).Curve()
        else:
            return GeomAPI_PointsToBSpline(u_points_array, deg_min, deg_max, GeomAbs_C2, precision).Curve()

    num_u_points = points.shape[0]
    u_points_array = TColgp_Array1OfPnt(1, num_u_points)
    for u_index in range(1, num_u_points + 1):
        pt = points[u_index - 1]
        point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        u_points_array.SetValue(u_index, point_2d)

    precision = [FITTING_TOLERANCE, FITTING_TOLERANCE * 2, FITTING_TOLERANCE * 5, FITTING_TOLERANCE * 10]
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


def create_wire_from_unordered_edges(face_edges, max_retry_times=10, is_sort_by_length=False):
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


def create_trimmed_face(surface, wire_list):
    face_fixer = ShapeFix_Face()
    face_fixer.Init(surface, CONNECT_TOLERANCE, True)
    for wire in wire_list:
        face_fixer.Add(wire)
    face_fixer.SetAutoCorrectPrecisionMode(False)
    face_fixer.SetPrecision(CONNECT_TOLERANCE)
    face_fixer.SetMaxTolerance(CONNECT_TOLERANCE)
    face_fixer.SetFixOrientationMode(True)
    face_fixer.SetFixSplitFaceMode(True)
    face_fixer.SetFixMissingSeamMode(True)
    face_fixer.SetFixWireMode(True)
    face_fixer.SetFixLoopWiresMode(True)
    face_fixer.SetFixIntersectingWiresMode(True)
    face_fixer.SetFixPeriodicDegeneratedMode(True)
    face_fixer.SetFixSmallAreaWireMode(True)
    face_fixer.SetRemoveSmallAreaFaceMode(True)
    face_fixer.Perform()

    face_fixer.FixMissingSeam()
    face_fixer.FixAddNaturalBound()
    face_fixer.FixOrientation()
    face_fixer.FixWiresTwoCoincEdges()
    face_fixer.FixIntersectingWires()
    face_fixer.FixPeriodicDegenerated()
    face_fixer.Perform()

    # wire_arr = TopTools_HSequenceOfShape()
    # for wire in wire_list:
    #     wire_arr.Append(wire)
    # face_fixer.FixLoopWire(wire_arr)
    face_occ = face_fixer.Face()
    return face_occ


def check_edges_in_face(face, edges, dist_tol=CONNECT_TOLERANCE):
    for edge in edges:
        edge_vertexes = get_edge_vertexes(edge)
        for vertex in edge_vertexes:
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, face)
            min_dist = dist_shape_shape.Value()
            if min_dist > dist_tol:
                return False
    return True


"""
Fit parametric surfaces / curves and trim into B-rep
"""


def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, folder_path, isdebug=False, is_save_face=True, debug_face_idx=[]):
    ##################################### 1. Fit primitives #####################################
    recon_faces = [create_surface(points) for points in surf_wcs]
    recon_curves = [create_edge(points) for points in edge_wcs]

    # Create edges from the curve list
    recon_edge = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in recon_curves]

    ###################################### 2. Trim Face ##########################################
    # Cut surface by wire
    is_face_success_list = []
    trimmed_faces = []
    for idx, (surface, face_edge_idx) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        face_edges = [recon_edge[edge_idx] for edge_idx in face_edge_idx]
        if idx in debug_face_idx:
            is_viz_wire, is_viz_face, is_viz_shell = True, True, True
        else:
            is_viz_wire, is_viz_face, is_viz_shell = False, False, False

        # 2. Construct wires from edges
        wire_list = create_wire_from_unordered_edges(face_edges)

        # visualize the constructed wire
        if is_viz_wire:
            viz_shapes(wire_list)

        # 3. Construct face using geom surface and wires
        trimmed_face = create_trimmed_face(surface, wire_list)
        trimmed_faces.append(trimmed_face)

        if is_viz_face:
            viz_shapes([trimmed_face])

        # check if the face contains all edges
        is_face_success = check_edges_in_face(trimmed_face, face_edges)
        is_face_success_list.append(is_face_success)

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
                write_stl_file(trimmed_face, os.path.join(folder_path, 'recon_face', f'{idx}.stl'), linear_deflection=0.001,
                               angular_deflection=0.5)
            except:
                print(f"Error writing step or stl file for face {idx}")

    ###################################### 3. Sew solid ##########################################
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(SEWING_TOLERANCE)
    # sewing.SetLocalTolerancesMode(True)
    for face in trimmed_faces:
        sewing.Add(face)
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    if isdebug and is_viz_shell:  # sewn_shell.ShapeType() == TopAbs_COMPOUND:
        viz_shapes([sewn_shell])

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

    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    fix_solid = ShapeFix_Solid(solid)
    fix_solid.SetPrecision(FIX_TOLERANCE)
    fix_solid.SetMaxTolerance(FIX_TOLERANCE)
    fix_solid.SetFixShellMode(True)
    fix_solid.SetFixShellOrientationMode(True)
    fix_solid.SetCreateOpenSolidMode(True)
    fix_solid.Perform()
    fixed_solid = fix_solid.Solid()

    return fixed_solid, is_face_success_list
