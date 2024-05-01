import os
from pathlib import Path
import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.GeomAbs import (GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_Ellipse,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_C2)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file

from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Display.SimpleGui import init_display

root = Path("G:/Dataset/img2brep/test_temp_0430")

if __name__ == '__main__':
    for prefix in os.listdir(root):
        prefix_path = root / prefix
        data = np.load(root / prefix / "data.npz")

        gt_faces = data["gt_face"].astype(np.float64)
        gt_edges = data["gt_edge_point"].astype(np.float64)

        num_points_u = 20  # Number of points in U-direction
        num_points_v = 20  # Number of points in V-direction
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.View.Camera().SetProjectionType(1)

        b_spline_faces = []
        for face in gt_faces:
            # Construct a BSpline in OCCT
            points = TColgp_Array2OfPnt(1, num_points_u, 1, num_points_v)
            for i in range(1, num_points_u + 1):
                for j in range(1, num_points_v + 1):
                    points.SetValue(i, j, gp_Pnt(face[i - 1, j - 1, 0], face[i - 1, j - 1, 1], face[i - 1, j - 1, 2]))

            surface_builder = GeomAPI_PointsToBSplineSurface(points, 3, 8, GeomAbs_C2, 0.001)
            bspline_surface = surface_builder.Surface()

            display.DisplayShape(bspline_surface, update=True)

            # Check if the surface was created successfully
            if surface_builder.IsDone():
                print("BSpline surface created successfully.")
            else:
                print("Failed to create BSpline surface.")
            b_spline_faces.append(bspline_surface)

        b_spline_curves = []
        for edge in gt_edges:
            # Construct a BSpline in OCCT
            points = TColgp_Array1OfPnt(1, num_points_u)
            for i in range(1, num_points_u + 1):
                for j in range(1, num_points_v + 1):
                    points.SetValue(i, gp_Pnt(edge[i - 1, 0], edge[i - 1, 1], edge[i - 1, 2]))

            curve_builder = GeomAPI_PointsToBSpline(points, 3, 8, GeomAbs_C2, 0.001)
            bspline_curve = curve_builder.Curve()

            display.DisplayShape(bspline_curve, update=True)

            # Check if the surface was created successfully
            if curve_builder.IsDone():
                print("BSpline curve created successfully.")
            else:
                print("Failed to create BSpline curve.")
            b_spline_curves.append(bspline_curve)



        start_display()

        pass
