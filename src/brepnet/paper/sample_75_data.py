from pathlib import Path
from src.brepnet.eval.check_valid import *
import random, trimesh
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

from src.brepnet.post.utils import triangulate_shape

brepgen_root = Path("/mnt/d/uncond_results/1203/1127_730_li_270k_1gpu_post/")
out_root = Path("/mnt/d/uncond_results/1203/1127_730_li_270k_1gpu_75")
out_root.mkdir(exist_ok=True, parents=True)

all_folders = list(brepgen_root.iterdir())
random.seed(0)
random.shuffle(all_folders)
shapedir = all_folders[:75]

def to_mesh(face_points):
    num_u_points, num_v_points = face_points.shape[1], face_points.shape[2]
    mesh_total = trimesh.Trimesh()
    for idx in range(face_points.shape[0]):
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1, num_u_points + 1):
            for v_index in range(1, num_v_points + 1):
                pt = face_points[idx][u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)

        approx_face = GeomAPI_PointsToBSplineSurface(
                uv_points_array, 3, 8,
                GeomAbs_C2, 5e-2).Surface()

        v, f = triangulate_shape(BRepBuilderAPI_MakeFace(approx_face, 5e-2).Face())
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        mesh_total += mesh_item
    return mesh_total

for item in tqdm(shapedir):
    idx = item.name
    if (item/"recon_brep.step").exists() and check_step_valid_soild(item/"recon_brep.step"):
        shutil.copy(item/"recon_brep.step", out_root/f"{idx}.step")
    else:
        if False:
            npz_data = np.load(item/"data_src.npz")
        else:
            npz_data = np.load(brepgen_root / "../" / brepgen_root.name[:-5] / idx / "data.npz")
        pred_face = npz_data["pred_face"]
        mesh_item = to_mesh(pred_face)
        (out_root/idx).mkdir(exist_ok=True, parents=True)
        mesh_item.export(out_root/idx/f"{idx}_model.obj")
        
        with open(str(out_root/idx/f"{idx}_wire.obj"), 'w') as wire_file:
            vertex_index = 1
            for edge in npz_data["pred_edge"]:
                for v in edge:
                    wire_file.write(f"v {v[0]} {v[1]} {v[2]}\n")

                for i in range(vertex_index, vertex_index + 16 - 1):
                    wire_file.write(f"l {i} {i + 1}\n")

                vertex_index += 16
        
        with open(str(out_root/idx/f"{idx}_vertex.obj"), 'w') as vertex_file:
            for e in npz_data["pred_edge"]:
                vertex_file.write(f"v {e[0][0]} {e[0][1]} {e[0][2]}\n")
                vertex_file.write(f"v {e[-1][0]} {e[-1][1]} {e[-1][2]}\n")
