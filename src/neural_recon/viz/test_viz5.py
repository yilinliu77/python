import os

from torch.utils.data import DataLoader

import cv2
import numpy as np
import torch
from shared.common_utils import to_homogeneous_tensor

from src.neural_recon.phase5 import prepare_dataset_and_model, LModel21, Multi_node_single_img_dataset

if __name__ == '__main__':
    # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("1", 1600, 900)
    # cv2.moveWindow("1", 0, 0)

    id_viz_face = [14]

    data = prepare_dataset_and_model("D:/DATASET/SIGA2023/Mechanism/ABC-NEF-COLMAP/00000077",
                                     id_viz_face,
                                     [0., 0., 0., 1., 1., 1.])

    model = LModel21(data,
                     False,
                     id_viz_face,
                     "imgs_log"
                     )

    state_dicts = torch.load("outputs/2023_05_05_14_59_40/lightning_logs/version_0/checkpoints/epoch=19-step=60.ckpt")["state_dict"]
    state_dicts = {key[6:]:state_dicts[key] for key in state_dicts}
    model.load_state_dict(state_dicts)
    # model.is_regress_normal=True
    model.eval()
    model.cuda()

    dataset = Multi_node_single_img_dataset(data, False, id_viz_face, "training")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
               collate_fn=Multi_node_single_img_dataset.collate_fn,
               num_workers=0)

    with torch.no_grad():
        big_img1 = None
        big_img2 = []

        for id_vertice, idxs_ in enumerate((dataloader)):
            idxs=[]
            for item in idxs_:
                idxs.append(item.cuda())
            is_single_img = True
            # 0: Unpack data
            # (1,)
            id_cur_imgs = idxs[0]
            # (B, E, 4)
            batched_ids = idxs[1]
            # (N, 4, 4)
            transformations = idxs[2]
            # (3, 3)
            intrinsic = idxs[3]
            # (N+1, h, w)
            imgs = idxs[4]
            # (B, E * 2, 3)
            ray_c = idxs[5]
            # (B, E * 2)
            valid_flags = idxs[6]
            id_vertices = idxs[7]
            batch_size = batched_ids.shape[0]
            num_vertices = batch_size
            num_max_edges_per_vertice = batched_ids.shape[1]
            device = id_cur_imgs.device

            # (B * E)
            id_start_point = batched_ids[:, :, 0]
            id_start_point_unique = batched_ids[:, 0, 0]
            # (B, E)
            id_end_point = batched_ids[:, :, 1]
            # (B, E, 3)
            start_ray_c = ray_c[:, ::2]
            # (B, E, 3)
            end_ray_c = ray_c[:, 1::2]

            # (B, E)
            start_point_distances = model.distances[id_cur_imgs][id_start_point,0]
            end_point_distances = model.distances[id_cur_imgs][id_end_point,0]
            # (B, E, 3)
            end_points_c = end_ray_c * end_point_distances[:, :, None] * model.distance_normalizer
            start_points_c = start_ray_c * start_point_distances[:, :, None] * model.distance_normalizer

            # Make different combinations of the edges
            num_imgs = transformations.shape[0]
            num_points = start_points_c.shape[0]
            num_max_edges = start_points_c.shape[1]
            similarity_loss, similarity_mask, black_area_in_img1s, [p1,p2] = model.compute_similarity_wrapper(
                start_points_c.reshape(-1, 3),
                end_points_c.reshape(-1, 3),
                imgs, transformations, intrinsic
            )

            # Unpack the combination
            similarity_loss = similarity_loss.reshape((num_imgs, num_points, num_max_edges))  # 4 combinations
            similarity_mask = similarity_mask.reshape((num_imgs, num_points, num_max_edges))
            black_area_in_img1s = black_area_in_img1s.reshape((num_points, num_max_edges))
            valid_flags = torch.logical_and(~black_area_in_img1s, valid_flags)
            penalized_loss = 10.
            # penalized_loss = torch.inf
            similarity_loss[~similarity_mask] = penalized_loss
            similarity_loss = similarity_loss.permute(1, 2, 0)
            similarity_loss[~valid_flags] = penalized_loss
            if is_single_img:
                similarity_loss_avg = similarity_loss[..., 1]
            else:
                similarity_loss_avg = torch.mean(similarity_loss, dim=3)
            final_loss = torch.mean(similarity_loss_avg)

            # LOG
            log_str = "{:02d}: Flag:".format(id_vertice,)
            for i in range(valid_flags.shape[1]):
                log_str+="{:5},".format(str(valid_flags[0,i].item()))
            log_str+="; Loss:"
            for i in range(valid_flags.shape[1]):
                log_str+="{:6.3f},".format(similarity_loss_avg[0,i].item())
            print(log_str)
            rgb1 = cv2.cvtColor((imgs[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            shape1 = rgb1.shape[:2][::-1]

            start_points_c = start_points_c.reshape(-1, 3)
            end_points_c = end_points_c.reshape(-1, 3)
            start_points_c = start_points_c[(start_points_c != 0).all(axis=1)]
            end_points_c = end_points_c[(end_points_c != 0).all(axis=1)]
            start_points_2d1 = (intrinsic @ start_points_c.T).T
            start_points_2d1 = (start_points_2d1[:, :2] / start_points_2d1[:, 2:3]).cpu().numpy()
            start_points_2d1 = (np.clip(start_points_2d1, 0, 0.99999) * shape1).astype(int)
            end_points_2d1 = (intrinsic @ end_points_c.T).T
            end_points_2d1 = (end_points_2d1[:, :2] / end_points_2d1[:, 2:3]).cpu().numpy()
            end_points_2d1 = (np.clip(end_points_2d1, 0, 0.99999) * shape1).astype(int)


            line_img1 = rgb1.copy()
            if big_img1 is None:
                big_img1 = line_img1.copy()

            line_thickness = 1
            point_thickness = 2
            point_radius = 1

            p1 = (np.clip(p1.cpu().numpy(), 0, 0.99999) * shape1).astype(int)
            line_img1[p1[:,1],p1[:,0]] = (0,0,255)
            for id_ver, _ in enumerate(end_points_2d1):
                if valid_flags[0, id_ver] and similarity_loss_avg[0, id_ver] < 1e-1:
                    cv2.line(big_img1, start_points_2d1[id_ver], end_points_2d1[id_ver], (0, 0, 255),
                             thickness=line_thickness)
                cv2.line(line_img1, start_points_2d1[id_ver], end_points_2d1[id_ver], (0, 0, 255),
                         thickness=line_thickness)

            for id_ver, _ in enumerate(end_points_2d1):
                if valid_flags[0,id_ver] and similarity_loss_avg[0, id_ver] < 1e-1:
                    cv2.circle(big_img1, start_points_2d1[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(big_img1, end_points_2d1[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                cv2.circle(line_img1, start_points_2d1[id_ver], radius=point_radius,
                           color=(0, 255, 255), thickness=point_thickness)
                cv2.circle(line_img1, end_points_2d1[id_ver], radius=point_radius,
                           color=(0, 255, 255), thickness=point_thickness)

            line_imgs2 = []
            for i_img in range(imgs[1:].shape[0]):
                rgb2 = cv2.cvtColor((imgs[1 + i_img].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                line_img2 = rgb2.copy()
                if len(big_img2) < 10:
                    big_img2.append(line_img2.copy())
                shape2 = rgb2.shape[:2][::-1]
                start_points_2d2 = (transformations[i_img] @ to_homogeneous_tensor(start_points_c).T).T
                start_points_2d2 = (start_points_2d2[:, :2] / start_points_2d2[:, 2:3]).cpu().numpy()
                start_points_2d2 = (np.clip(start_points_2d2, 0, 0.99999) * shape2).astype(int)
                end_points_2d2 = (transformations[i_img] @ to_homogeneous_tensor(end_points_c).T).T
                end_points_2d2 = (end_points_2d2[:, :2] / end_points_2d2[:, 2:3]).cpu().numpy()
                end_points_2d2 = (np.clip(end_points_2d2, 0, 0.99999) * shape2).astype(int)

                for id_ver, _ in enumerate(end_points_2d2):
                    if valid_flags[0, id_ver] and similarity_loss_avg[0, id_ver] < 1e-1:
                        cv2.line(big_img2[i_img], start_points_2d2[id_ver], end_points_2d2[id_ver], (0, 0, 255),
                                 thickness=line_thickness)
                    cv2.line(line_img2, start_points_2d2[id_ver], end_points_2d2[id_ver], (0, 0, 255),
                             thickness=line_thickness)
                for id_ver, _ in enumerate(end_points_2d2):
                    if valid_flags[0, id_ver] and similarity_loss_avg[0, id_ver] < 1e-1:
                        cv2.circle(big_img2[i_img], start_points_2d2[id_ver], radius=point_radius,
                                   color=(0, 255, 255), thickness=point_thickness)
                        cv2.circle(big_img2[i_img], end_points_2d2[id_ver], radius=point_radius,
                                   color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(line_img2, start_points_2d2[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(line_img2, end_points_2d2[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                line_imgs2.append(line_img2)

            big_imgs = np.concatenate(
                (np.concatenate(
                    (line_img1, line_imgs2[0], line_imgs2[1], line_imgs2[2]), axis=1),
                 np.concatenate(
                     (line_imgs2[3], line_imgs2[4], line_imgs2[5], line_imgs2[6]), axis=1),
                 np.concatenate(
                     (line_imgs2[7], line_imgs2[8], line_imgs2[9], line_imgs2[9]), axis=1),
                )
                , axis=0)

            fontFace = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 1.5
            thickness = 1
            cv2.putText(big_imgs, log_str, (50, 100), fontFace, fontScale, (0, 0, 255), thickness)
            cv2.imwrite(os.path.join("outputs/test_viz_5", "2d_{:05d}.jpg".format(id_vertice)), big_imgs)

        big_imgs = np.concatenate(
            (np.concatenate(
                (big_img1, big_img2[0], big_img2[1], big_img2[2]), axis=1),
             np.concatenate(
                 (big_img2[3], big_img2[4], big_img2[5], big_img2[6]), axis=1),
             np.concatenate(
                 (big_img2[7], big_img2[8], big_img2[9], big_img2[9]), axis=1),
            )
            , axis=0)
        cv2.imwrite(os.path.join("outputs/test_viz_5", "xxx.jpg".format(id_vertice)), big_imgs)

        pass
