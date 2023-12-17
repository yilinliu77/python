import os

import numpy as np
import torch
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.structures import Meshes
# from lfd import LightFieldDistance
from tqdm import tqdm
import pytorch3d
import open3d as o3d

pred_dir = r"G:/Dataset/GSP/test_data_small/extracted_mesh_v14"
gt_dir = r"G:/Dataset/GSP/test_data_small/mesh"

if __name__ == '__main__':
    files = sorted(os.listdir(gt_dir))

    ncs=[]
    for file in tqdm(files):
        prefix = file[:8]
        pred_model_path = os.path.join(pred_dir, prefix, "mesh", "0total_mesh.ply")
        if not os.path.exists(pred_model_path):
            print("Not exist: ", pred_model_path)
            continue

        mesh = o3d.io.read_triangle_mesh(pred_model_path)

        with torch.no_grad():
            verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
            faces_idx = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32)

            trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
            nc = pytorch3d.loss.mesh_normal_consistency(trg_mesh)

        ncs.append(nc.item())

        # lfd_value: float = LightFieldDistance(verbose=True).get_distance(
        #     pred_model.vertices, pred_model.faces,
        #     gt_model.vertices, gt_model.faces
        # )
        # print("{}: {}".format(prefix, nc))
        pass

    print("Average NC: ", np.mean(ncs))