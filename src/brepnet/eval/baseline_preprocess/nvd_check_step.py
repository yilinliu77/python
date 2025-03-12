import os

from tqdm import tqdm
import trimesh
import argparse

from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse)
from shared.occ_utils import get_primitives

from src.brepnet.eval.check_valid import check_step_valid_soild

def is_step_valid(step_path):
    assert os.path.exists(step_path)
    _, step_shape = check_step_valid_soild(step_path, return_shape=True)

    exp = TopExp_Explorer(step_shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        surface_adaptor = BRepAdaptor_Surface(face)
        surface_type = surface_adaptor.GetType()
        if surface_type not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere]:
            # print(surface_type)
            return False
        exp.Next()
    exp = TopExp_Explorer(step_shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_adaptor = BRepAdaptor_Curve(edge)
        curve_type = curve_adaptor.GetType()
        if curve_type not in [GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse]:
            # print(curve_type)
            return False
        exp.Next()
    return True

# E:\data\img2brep\data\deepcad_v6_test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_txt', type=str)
    args = parser.parse_args()

    data_root = args.data_root
    assert os.path.exists(data_root), f"Data root not exists: {data_root}"
    if not args.output_txt:
        args.output_txt = os.path.join(os.path.dirname(data_root),
                                       os.path.basename(data_root) + "_valid.txt")
    output_dir = os.path.dirname(args.data_root)
    os.makedirs(output_dir, exist_ok=True)
    txt_fp = open(args.output_txt, 'w')

    all_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    is_valid_step_folder = []
    for folder in tqdm(all_folders):
        step_path = os.path.join(data_root, folder, "normalized_shape.step")
        is_valid = is_step_valid(step_path)
        print(folder, is_valid)
        if is_valid:
            is_valid_step_folder.append(folder)
            txt_fp.write(f'{folder}\n')
    txt_fp.close()

    print(f"Total valid step files: {len(is_valid_step_folder)}")