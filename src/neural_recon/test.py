import random
import sys, os

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("thirdparty/sdf_computer/build/")
from src.neural_recon.phase2 import Phase2

if __name__ == '__main__':
    model = Phase2({
        "dataset":
            {"scene_boundary": [-50, -50, -10, 250, 200, 60],
             "colmap_dir": "d:/Projects/NeuralRecon/Test_data/OBL_L7/Test_imgs2_colmap_neural/sparse_align",
             "img_nif_dir": "output/neural_recon/img_nif_log",
             "img_size": [600, 400],
             "num_sample": [300000000, 300000000, 300000000],
             "resample_after_n_epoches": 100},
        "trainer":
            {"check_val_every_n_epoch": 100,
             "learning_rate": 1e-4,
             "gpu": 1,
             "num_worker": 0,
             "batch_size": 262144,
             "auto_lr_find": "false",
             "evaluate": "false",
             "running_sanity_check": "false",
             "output": "neural_recon"}
    })

    cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1",1600,900)
    cv2.moveWindow("1", 0, 0)
    variances = [[],[],[],[]]
    ids = np.random.randint(0, 2000000, 10000)
    with torch.no_grad():
        for idx in tqdm(ids):
            # Sample point
            sample_point = model.data["sample_points"][idx]

            # Camera
            visibility = model.data["final_visibility"][:,idx]
            imgs = [item for id,item in enumerate(model.data["img_database"]) if visibility[id]]
            print("Sample point: {}; SDF: {}".format((sample_point-0.5) * model.bounds_size+model.bounds_center, model.data["sample_distances"][idx]))
            predicted_rgbs=[]
            gt_rgbs=[]
            for item in imgs:
                # img = cv2.imread(item.img_path, cv2.IMREAD_UNCHANGED)
                # img = cv2.resize(img, (600,400), cv2.INTER_AREA)
                projection = item.projection
                pixel_pos = projection @ np.insert(sample_point, 3, 1)
                pixel_pos = pixel_pos[:2] / pixel_pos[2]
                predicted_rgb = model.img_models[item.img_name](torch.from_numpy(pixel_pos).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0))
                predicted_rgbs.append(predicted_rgb)
                pixel_pos = (pixel_pos*np.array([600,400])).astype(np.int)
                # gt_rgb = img[pixel_pos[1],pixel_pos[0]] / 255.
                # gt_rgbs.append(gt_rgb)
                # print(predicted_rgb.detach().cpu().numpy()[0])
                # viz_img = cv2.circle(img,pixel_pos, 10, (0,0,255), 5)
                # cv2.imshow("1",viz_img)
                # cv2.waitKey()
            if len(predicted_rgbs) <= 1:
                continue
            predicted_variance = torch.var(torch.concatenate(predicted_rgbs,dim=0)).cpu().item()
            # gt_variance = np.var(np.concatenate(gt_rgbs,axis=0))
            # print("Predicted Variance: {}; GT Variance: {}".format(predicted_variance, gt_variance))
            if idx>1000000:
                variances[2].append(predicted_variance)
                # variances[3].append(gt_variance)
            else:
                variances[0].append(predicted_variance)
                # variances[1].append(gt_variance)
            pass
        print(np.mean(variances[0]))
        # print(np.mean(variances[1]))
        print(np.mean(variances[2]))
        # print(np.mean(variances[3]))
        pass