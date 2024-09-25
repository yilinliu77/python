import os
from pathlib import Path

import numpy as np
import trimesh


def export(v_points, f_file, v_color=None):
    trimesh.PointCloud(v_points.reshape(-1,3), colors=v_color).export(f_file)

root=Path(r"D:/img2brep/0924_0921_pure_test")
output_root=Path(r"D:/img2brep/temp")

if __name__ == '__main__':
    output_root.mkdir(parents=True, exist_ok=True)
    files = os.listdir(root)
    files.sort()
    for file in files[:10]:
        if file.endswith("feature.npz"):
            continue
        # data = np.load(root/file, allow_pickle=True)["arr_0"][()]
        data = np.load(root/file, allow_pickle=True)
        pred_face_points = data['pred_face']
        pred_edge_points = data['pred_edge']

        gt_face_points = data['gt_face']
        gt_edge_points = data['gt_edge']

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

