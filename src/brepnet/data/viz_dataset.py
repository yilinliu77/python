import os

import numpy as np
from OCC.Core.Aspect import Aspect_DisplayConnection
from OCC.Core.Graphic3d import Graphic3d_NameOfMaterial_Gold, Graphic3d_NameOfMaterial_Steel, Graphic3d_MaterialAspect, \
    Graphic3d_NameOfMaterial_Aluminum, Graphic3d_NameOfMaterial_Silver, Graphic3d_NameOfMaterial
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.V3d import V3d_Viewer, V3d_View
from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir, gp_Vec
from OCC.Display.SimpleGui import init_display
from OCC.Core.AIS import AIS_Shape, AIS_Shaded, AIS_WireFrame
from OCC.Extend.DataExchange import read_step_file

from shared.occ_utils import normalize_shape

import gurobipy as gp

radius = 4
views = [
        gp_Pnt(-2, -2, -2),
        gp_Pnt(-2, -2, 0),
        gp_Pnt(-2, -2, 2),
        gp_Pnt(-2, 0, -2),
        gp_Pnt(-2, 0, 0),
        gp_Pnt(-2, 0, 2),
        gp_Pnt(-2, 2, -2),
        gp_Pnt(-2, 2, 0),
        gp_Pnt(-2, 2, 2),

        gp_Pnt(0, -2, -2),
        gp_Pnt(0, -2, 0),
        gp_Pnt(0, -2, 2),
        gp_Pnt(0, 2, -2),
        gp_Pnt(0, 2, 0),
        gp_Pnt(0, 2, 2),

        gp_Pnt(2, -2, -2),
        gp_Pnt(2, -2, 0),
        gp_Pnt(2, -2, 2),
        gp_Pnt(2, 0, -2),
        gp_Pnt(2, 0, 0),
        gp_Pnt(2, 0, 2),
        gp_Pnt(2, 2, -2),
        gp_Pnt(2, 2, 0),
        gp_Pnt(2, 2, 2),
    ]

if __name__ == '__main__':
    gp.Model()
    shape = read_step_file(r"D:/Datasets/00000003/00000003_1ffb81a71e5b402e966b9341_step_002.step", verbosity=False)

    shape = normalize_shape(shape, 0.9)

    display, start_display, add_menu, add_function_to_menu = init_display(
        size=(256,256),
        display_triedron = False,
        background_gradient_color1=[255, 255, 255],
        background_gradient_color2=[255, 255, 255],
    )
    display.camera.SetProjectionType(1)
    display.View.TriedronErase()
    ais_shape = AIS_Shape(shape)
    ais_shape.SetMaterial(Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_Silver))
    display.Context.Display(ais_shape, True)
    start_display()
    display.View.Dump(f"view_{-1}.png")
    # display.FitAll()
    for i, view in enumerate(views):
        display.camera.SetEyeAndCenter(gp_Pnt(view.X(), view.Y(), view.Z()), gp_Pnt(0., 0., 0.))
        display.camera.SetDistance(radius)
        display.camera.SetUp(gp_Dir(0,0,1))
        display.camera.SetAspect(1)
        display.camera.SetFOVy(45)
        filename = f"view_{i}.png"
        display.View.Dump(filename)

