import math
import random
import sys, os


sys.path.append("thirdparty/sdf_computer/build/")

import cv2
import numpy as np
import torch
from tqdm import tqdm

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
    from src.neural_recon.phase2 import Phase2
    from src.neural_recon.colmap_dataset import Blender_Segment_dataset
    model = Phase2({
        "dataset":
            {"scene_boundary": [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
             "colmap_dir": "C:/Users/whats/Dropbox/Project/2022-NeuralStructure/Illustrations/Cube",
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
        "img_database": model.raw_data,
        "segments": np.asarray([
            [[0.984628, -0.414957, 0.204155], [0.997870, 0.414072, 0.205961]],
            [[1, -1, 1], [1, -1, -1]],
            [[1.5, -0.414957, 0.204155], [1.5, 0.414072, 0.205961]],
        ]),
        "segments_visibility": np.asarray([
            True,
            True,
            True,
        ])[np.newaxis, :].repeat(len(model.raw_data), 0)
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
                                   (0, 0, 255), 2)
                viz_imgs.append(viz_img)
            viz_imgs = np.concatenate(viz_imgs, axis=1)
            print("==========={}=============".format(idx))
            print(data_item["ncc"])
            print(data_item["edge_similarity"])
            print(data_item["edge_magnitude"])
            cv2.imshow("1", viz_imgs)
            cv2.waitKey()
            pass
        pass

if __name__ == '__main__2':
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    img1 = cv2.imread(r"C:\Users\whats\Dropbox\Project\2022-NeuralStructure\Illustrations\Cube\0.png")
    img2 = cv2.imread(r"C:\Users\whats\Dropbox\Project\2022-NeuralStructure\Illustrations\Cube\1.png")

    segment1 = np.array([
        [946, 451, 946, 909],
        [946, 451, 1307, 307],
        [1004, 699, 1004, 803],
    ], dtype=np.int64)
    segment2 = np.array([
        [547, 363, 582, 790],
        [547, 363, 950, 524],
        [626, 650, 630, 748],
    ], dtype=np.int64)

    keylines1 = keyline_from_seg(segment1)
    keylines2 = keyline_from_seg(segment2)

    lsd_detector = cv2.line_descriptor.LSDDetector.createLSDDetector()
    lsd_key_lines_single_scale1 = lsd_detector.detect(img1, 1, 1)
    lsd_key_lines_multi_scale1 = lsd_detector.detect(img1, 2, 3)
    lsd_key_lines_single_scale2 = lsd_detector.detect(img2, 1, 1)
    lsd_key_lines_multi_scale2 = lsd_detector.detect(img2, 2, 3)

    descriptor = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
    lsd_descriptor_single_scale1 = descriptor.compute(img1, lsd_key_lines_single_scale1)[1]
    lsd_descriptor_multi_scale1 = descriptor.compute(img1, lsd_key_lines_multi_scale1)[1]
    my_descriptor1 = descriptor.compute(img1, keylines1)[1]
    lsd_descriptor_single_scale2 = descriptor.compute(img2, lsd_key_lines_single_scale2)[1]
    lsd_descriptor_multi_scale2 = descriptor.compute(img2, lsd_key_lines_multi_scale2)[1]
    my_descriptor2 = descriptor.compute(img2, keylines2)[1]

    # bdm = cv2.line_descriptor.BinaryDescriptorMatcher()
    # match_result = bdm.match(lsd_descriptor_multi_scale1, lsd_descriptor_multi_scale2)
    # for item in match_result:
    #     viz_img1 = np.copy(img1)
    #     viz_img2 = np.copy(img2)
    #     cv2.line(viz_img1,
    #              (int(lsd_key_lines_multi_scale1[item.queryIdx].startPointX),int(lsd_key_lines_multi_scale1[item.queryIdx].startPointY)),
    #              (int(lsd_key_lines_multi_scale1[item.queryIdx].endPointX),int(lsd_key_lines_multi_scale1[item.queryIdx].endPointY)),
    #              (0, 0, 255), 2)
    #     cv2.line(viz_img2,
    #              (int(lsd_key_lines_multi_scale2[item.trainIdx].startPointX),int(lsd_key_lines_multi_scale2[item.trainIdx].startPointY)),
    #              (int(lsd_key_lines_multi_scale2[item.trainIdx].endPointX),int(lsd_key_lines_multi_scale2[item.trainIdx].endPointY)),
    #              (0, 0, 255), 2)
    #     viz_img = np.concatenate([viz_img1, viz_img2], axis=1)
    #     cv2.imshow("1", viz_img)
    #     cv2.waitKey()

    for i in range(3):
        cv2.line(img1,
                 segment1[i, 0],
                 segment1[i, 1],
                 (0, 0, 255), 2)
        cv2.line(img2,
                 segment2[i, 2],
                 segment2[i, 3],
                 (0, 0, 255), 2)
    viz_img = np.concatenate([img1, img2], axis=1)
    print(0)
    cv2.imshow("1", viz_img)
    cv2.waitKey()
    pass
