import json
import os
import shutil
from copy import deepcopy

import cv2
import hydra
import torch
from flask import Flask, request
from pytorch_lightning import seed_everything
from train import Mono_det_3d
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

app = Flask(__name__)

checkpoint_path = ""
global_id = 0
model = None


@hydra.main(config_name=".")
def init(v_cfg):
    # print(OmegaConf.to_yaml(v_cfg))
    # set_start_method('spawn')

    global model
    model = Mono_det_3d(v_cfg)
    model.load_state_dict(torch.load(v_cfg["trainer"]["resume_from_checkpoint"])["state_dict"], strict=True)
    model.eval()
    model.cuda()
    if os.path.exists("temp/viz_result"):
        shutil.rmtree("temp/viz_result")
    os.mkdir("temp/viz_result")


@app.route("/index", methods=["GET", "POST"])
def main():
    a = request.data
    print(request.args)
    img = np.fromstring(a, np.uint8).reshape(800, 800, 3)
    # cv2.imshow("1", img)
    # cv2.waitKey()
    global model, global_id

    """
    Img transform
    """
    with torch.no_grad():
        P2 = np.asarray([
            [802.275879, 0, 400, 0],
            [0, 802.275879, 400, 0],
            [0, 0, 1, 0],
        ]).astype(np.float32)
        img_tr, P2_tr = model.model.test_preprocess(img, p2=deepcopy(P2))

        img_tr = torch.from_numpy(img_tr).permute(2, 0, 1).unsqueeze(0).cuda()
        features = model.model.core(img_tr)
        cls_preds = model.model.cls_feature_extraction(features)
        reg_preds = model.model.reg_feature_extraction(features)

        scores, bboxes, cls_indexes = model.model.get_boxes(cls_preds, reg_preds,
                                                            model.model.anchors.to(cls_preds.device),
                                                            model.model.anchors_distribution.to(
                                                                cls_preds.device),
                                                            None,v_nms_threshold=0.3)
        valid_mask = scores > .75

        scores[valid_mask]
        bboxes[valid_mask]

        boxes = bboxes[valid_mask].cpu().numpy()

        if True:
            total_mesh = o3d.geometry.TriangleMesh()
            viz_img = deepcopy(img)
            for box in boxes:
                bbox_2d = model.model.rectify_2d_box(box[np.newaxis, :4], P2, P2_tr)
                pts = list(map(int, bbox_2d[0, 0:4]))
                viz_img = cv2.rectangle(viz_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 4)
                mesh = o3d.geometry.TriangleMesh.create_box(
                    width=max(0.1, box[11]),
                    height=max(0.1, box[10]),
                    depth=max(0.1, box[9]))
                mesh = mesh.translate([-box[11] / 2, -box[10] / 2, -box[9] / 2])

                mesh = mesh.rotate(R.from_euler("XYZ", [0, box[12], 0]).as_matrix())
                mesh = mesh.translate(box[6:9])
                total_mesh += mesh
            o3d.io.write_triangle_mesh("temp/viz_result/{}.ply".format(global_id), total_mesh)
            viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("temp/viz_result/{}.png".format(global_id), viz_img)

    return_results = {}
    for id, box in enumerate(np.concatenate([boxes,scores[valid_mask].cpu().numpy()[:,np.newaxis]],axis=1)):
        return_results[str(id)] = box.tolist()
    global_id += 1
    return json.dumps(return_results)


if __name__ == '__main__':
    seed_everything(0)
    init()
    app.run(host="0.0.0.0")
