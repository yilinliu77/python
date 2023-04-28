import json
import os, sys

import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ops = os.path.join

root_path = r"D:/dataset/SIGA2023/Mechanism/"
output_path = r"D:/dataset/SIGA2023/Mechanism/ABC-NEF-COLMAP"

if __name__ == '__main__':
    for item_name in tqdm(os.listdir(os.path.join(root_path, "ABC-NEF"))):
        item_input_path = os.path.join(root_path, "ABC-NEF", item_name)
        item_output_path = os.path.join(output_path, item_name)
        os.makedirs(item_output_path, exist_ok=True)
        os.makedirs(ops(item_output_path, "imgs"), exist_ok=True)
        data = json.loads(open(os.path.join(item_input_path, "transforms_train.json")).read())

        intrinsics = data["frames"][0]["camera_intrinsics"]
        with open(os.path.join(item_output_path, "cameras.txt"), "w") as f:
            f.write("1 SIMPLE_PINHOLE 800 800 {} {} {}\n".format(intrinsics[0][0], intrinsics[0][2], intrinsics[1][2]))

        pose_str = ""
        for idx, frame in enumerate(data["frames"]):
            img_name = frame["file_path"].split("/")[-1] + ".png"
            img_file = os.path.join(item_input_path, "train_img", img_name)
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img =  np.around(img[:,:,:3]*(img[:,:,3:4]/255.)).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(ops(item_output_path, "imgs", img_name), img)
            scale_matrix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            extrinsic = np.asarray(frame["transform_matrix"])
            # extrinsic = np.linalg.inv(scale_matrix@extrinsic)
            extrinsic = scale_matrix@np.linalg.inv(extrinsic)
            pos = extrinsic[:3, 3]
            quaternion = Rotation.from_matrix(extrinsic[:3, :3]).as_quat()
            # quaternion = Rotation.from_matrix(extrinsic[:3, :3]).as_quat()
            pose_str += "{} {} {} {} {} {} {} {} 1 {}\n\n".format(idx + 1,
                                                                quaternion[3], quaternion[0], quaternion[1],
                                                                quaternion[2],
                                                                pos[0], pos[1], pos[2],
                                                                img_name
                                                                )
            pass
        with open(ops(item_output_path, "images.txt"), "w") as f:
            f.write(pose_str)
        open(ops(item_output_path, "points3D.txt"), "w")

        obj_name = list(filter(lambda item:item[:8]==item_name,os.listdir(ops(root_path, "ABC_NEF_obj/obj"))))[0]
        mesh = o3d.io.read_triangle_mesh(ops(root_path, "ABC_NEF_obj/obj",obj_name))
        bbox = mesh.get_axis_aligned_bounding_box()
        max_axis = (bbox.max_bound-bbox.min_bound).max()
        vertices = np.asarray(mesh.vertices)
        vertices = vertices/max_axis
        center_point = (vertices.min(axis=0)+vertices.max(axis=0))/2
        vertices = vertices+0.5-center_point
        mesh.vertices=o3d.utility.Vector3dVector(vertices)
        o3d.io.write_triangle_mesh(ops(item_output_path, "mesh.ply"), mesh)
        pass