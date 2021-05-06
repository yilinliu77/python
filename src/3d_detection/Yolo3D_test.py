import shutil
from copy import deepcopy

import cv2
import hydra
import torch
from PIL import Image
from easydict import EasyDict as edict
import os
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything

from train import Mono_det_3d

dataset_root=r"D:\DATASET\KITTI\3DDetection\data_object_image_2\training\image_2"

test_index=[
    "000073","000074","000075","000079","000080",
    "000076","000077","000078","000081","000089",
            ]
checkpoint_path = r"lightning_logs/version_50/checkpoints/epoch=13-step=12991.ckpt"
img_file = r"D:\DATASET\KITTI\3DDetection\data_object_image_2\training\image_2\000002.png"
# img_file = r"D:\Projects\SEAR\datasets\withAlpha\training\image_2\002375.png"


@hydra.main(config_name=".")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)
    # set_start_method('spawn')

    model = Mono_det_3d(v_cfg)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=True)
    model.eval()
    model.cuda()
    with torch.no_grad():

        if os.path.exists("temp/viz_result"):
            shutil.rmtree("temp/viz_result")
        os.mkdir("temp/viz_result")

        for index in test_index:
            """
            Img transform
            """
            img = np.asarray(Image.open(os.path.join(dataset_root,index+".png")))
            P2 = np.asarray([item for item in open(os.path.join(dataset_root,"../calib/",index+".txt")).readlines()][2].split(":")[1].split(" ")[1:]).astype(np.float32).reshape(3,4)
            img_tr,P2_tr = model.model.test_preprocess(img,p2=deepcopy(P2))

            img_tr = torch.from_numpy(img_tr).permute(2, 0, 1).unsqueeze(0).cuda()
            features = model.model.core(img_tr)
            cls_preds = model.model.cls_feature_extraction(features)
            reg_preds = model.model.reg_feature_extraction(features)

            scores, bboxes, cls_indexes = model.model.get_boxes(cls_preds, reg_preds,
                                                                model.model.anchors.to(cls_preds.device),
                                                                model.model.anchors_distribution.to(cls_preds.device),
                                                                None)
            valid_mask = scores > .75

            scores[valid_mask]
            bboxes[valid_mask]

            boxes = bboxes[valid_mask].cpu().numpy()

            total_mesh = o3d.geometry.TriangleMesh()
            viz_img = deepcopy(img)
            for box in boxes:
                bbox_2d=model.model.rectify_2d_box(box[np.newaxis,:4],P2,P2_tr)
                pts = list(map(int, bbox_2d[0,0:4]))
                viz_img = cv2.rectangle(viz_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 4)
                mesh = o3d.geometry.TriangleMesh.create_box(
                    width=max(0.1, box[11]),
                    height=max(0.1, box[10]),
                    depth=max(0.1, box[9]))
                mesh = mesh.translate([-box[11] / 2, -box[10] / 2, -box[9] / 2])

                mesh = mesh.rotate(R.from_euler("XYZ", [0, box[12], 0]).as_matrix())
                mesh = mesh.translate(box[6:9])
                total_mesh += mesh
            o3d.io.write_triangle_mesh("temp/viz_result/{}.ply".format(index), total_mesh)
            viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("temp/viz_result/{}.png".format(index), viz_img)

        return


if __name__ == '__main__':
    main()
