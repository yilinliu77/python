import os
from pathlib import Path

import numpy as np
import trimesh
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face, topods

from shared.common_utils import check_dir
from src.brepnet.post.utils import create_surface, triangulate_shape, export_edges


def export(v_points, f_file, v_color=None):
    trimesh.PointCloud(v_points.reshape(-1,3), colors=v_color).export(f_file)

root=Path(r"D:/img2brep/2024_09_22_21_57_44_0921_pure")
output_root=Path(r"D:/img2brep/temp")
prefix = "00005247"

if __name__ == '__main__':
    check_dir(output_root)
    files = os.listdir(root)
    files.sort()

    if prefix == "":
        files = files[:10]
    else:
        files = [prefix+".npz"]
    for file in files:
        if file.endswith("feature.npz"):
            continue
        # data = np.load(root/file, allow_pickle=True)["arr_0"][()]
        data = np.load(root/file, allow_pickle=True)
        pred_face_points = data['pred_face']
        pred_edge_points = data['pred_edge']

        gt_face_points = data['gt_face']
        gt_edge_points = data['gt_edge']

        vertices = []
        faces = []
        num_p = 0
        for face_points in gt_face_points:
            face = create_surface(face_points, True)
            face = BRepBuilderAPI_MakeFace(face, 1e-2).Face()
            v,f = triangulate_shape(face)
            vertices.append(v)
            faces.append(f + num_p)
            num_p += v.shape[0]
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        export_edges(gt_edge_points, output_root/f"{(root/file).stem}_gt_wire.obj")
        trimesh.Trimesh(vertices, faces).export(output_root/f"{(root/file).stem}_gt_face.obj")

        vertices = []
        faces = []
        num_p = 0
        for face_points in pred_face_points:
            face = create_surface(face_points, True)
            face = BRepBuilderAPI_MakeFace(face, 1e-2).Face()
            v,f = triangulate_shape(face)
            vertices.append(v)
            faces.append(f + num_p)
            num_p += v.shape[0]
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        export_edges(pred_edge_points, output_root/f"{(root/file).stem}_pred_wire.obj")
        trimesh.Trimesh(vertices, faces).export(output_root/f"{(root/file).stem}_pred_face.obj")

        red = np.array([255,0,0],dtype=np.uint8)[None]
        green = np.array([0,255,0],dtype=np.uint8)[None]

        viz_gt_face_points = gt_face_points.reshape(-1,3)
        viz_pred_face_points = pred_face_points.reshape(-1,3)
        face_colors = np.concatenate((
            red.repeat(viz_gt_face_points.shape[0], axis=0),
            green.repeat(viz_pred_face_points.shape[0], axis=0)
        ), axis=0)
        export(np.concatenate((viz_gt_face_points, viz_pred_face_points)),
               output_root/f"{(root/file).stem}_face.ply", face_colors)

        viz_gt_edge_points = gt_edge_points.reshape(-1,3)
        viz_pred_edge_points = pred_edge_points.reshape(-1,3)
        edge_colors = np.concatenate((
            red.repeat(viz_gt_edge_points.shape[0], axis=0),
            green.repeat(viz_pred_edge_points.shape[0], axis=0)
        ), axis=0)
        export(np.concatenate((viz_gt_edge_points, viz_pred_edge_points)),
                output_root/f"{(root/file).stem}_edge.ply", edge_colors)

