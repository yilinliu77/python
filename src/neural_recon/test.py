import random
import sys, os

sys.path.append("thirdparty/sdf_computer/build/")

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.neural_recon.colmap_dataset import Blender_Segment_dataset

from src.neural_recon.phase2 import Phase2

if __name__ == '__main__1':
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

    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)
    variances = [[], [], [], []]
    ids = np.random.randint(0, 2000000, 10000)
    with torch.no_grad():
        for idx in tqdm(ids):
            # Sample point
            sample_point = model.data["sample_points"][idx]

            # Camera
            visibility = model.data["final_visibility"][:, idx]
            imgs = [item for id, item in enumerate(model.data["img_database"]) if visibility[id]]
            print("Sample point: {}; SDF: {}".format((sample_point - 0.5) * model.bounds_size + model.bounds_center,
                                                     model.data["sample_distances"][idx]))
            predicted_rgbs = []
            gt_rgbs = []
            for item in imgs:
                # img = cv2.imread(item.img_path, cv2.IMREAD_UNCHANGED)
                # img = cv2.resize(img, (600,400), cv2.INTER_AREA)
                projection = item.projection
                pixel_pos = projection @ np.insert(sample_point, 3, 1)
                pixel_pos = pixel_pos[:2] / pixel_pos[2]
                predicted_rgb = model.img_models[item.img_name](
                    torch.from_numpy(pixel_pos).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0))
                predicted_rgbs.append(predicted_rgb)
                pixel_pos = (pixel_pos * np.array([600, 400])).astype(np.int)
                # gt_rgb = img[pixel_pos[1],pixel_pos[0]] / 255.
                # gt_rgbs.append(gt_rgb)
                # print(predicted_rgb.detach().cpu().numpy()[0])
                # viz_img = cv2.circle(img,pixel_pos, 10, (0,0,255), 5)
                # cv2.imshow("1",viz_img)
                # cv2.waitKey()
            if len(predicted_rgbs) <= 1:
                continue
            predicted_variance = torch.var(torch.concatenate(predicted_rgbs, dim=0)).cpu().item()
            # gt_variance = np.var(np.concatenate(gt_rgbs,axis=0))
            # print("Predicted Variance: {}; GT Variance: {}".format(predicted_variance, gt_variance))
            if idx > 1000000:
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

if __name__ == '__main__':
    model = Phase2({
        "dataset":
            {"scene_boundary": [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
             "colmap_dir": "C:/Users/whats/Dropbox/Project/2022-Recon/Illustrations/Cube",
             "img_nif_dir": "output/neural_recon/img_nif_log",
             "trained_img_size": [600, 400],
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

    data = {
        "img_database": model.data["img_database"],
        "segments": np.asarray([
            [[0.984628, -0.414957, 0.204155], [0.997870, 0.414072, 0.205961]],
            [[1, -1, 1], [1, -1, -1]],
            [[1.5, -0.414957, 0.204155], [1.5, 0.414072, 0.205961]],
        ]),
        "segments_visibility": np.asarray([
            True,
            True,
            True,
        ])[np.newaxis, :].repeat(len(model.data["img_database"]), 0)
    }
    data["segments"] = ((data["segments"] - model.bounds_center) / model.bounds_size + 0.5)
    dataset = Blender_Segment_dataset(data, model.imgs, "validation")
    imgs = model.data["img_database"]

    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    with torch.no_grad():
        for idx in range(len(dataset)):
            data_item = dataset[idx]
            sample_segment = data_item["sample_segment"]
            projected_coordinates = (data_item["projected_coordinates"] * np.array((600, 400))).astype(np.int64)
            viz_imgs = []
            for id_img, item in enumerate(imgs):
                img = cv2.imread(item.img_path, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (600, 400), cv2.INTER_AREA)

                viz_img = cv2.line(img,
                                   projected_coordinates[id_img][0],
                                   projected_coordinates[id_img][1],
                                   (0, 0, 255), 1)
                viz_imgs.append(viz_img)
            viz_imgs = np.concatenate(viz_imgs,axis=1)
            print("==========={}=============".format(idx))
            print(data_item["ncc"])
            print(data_item["edge_similarity"])
            print(data_item["edge_magnitude"])
            cv2.imshow("1", viz_imgs)
            cv2.waitKey()
            pass
        pass
