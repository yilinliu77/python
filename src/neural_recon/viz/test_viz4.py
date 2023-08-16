import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from shared.common_utils import to_homogeneous_tensor, pad_imgs

from src.neural_recon.bak.phase3 import prepare_dataset_and_model, LModel19

if __name__ == '__main__':
    # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("1", 1600, 900)
    # cv2.moveWindow("1", 0, 0)

    id_viz_face = 1476
    id_viz_edge = 9118

    data = prepare_dataset_and_model("d:/Projects/NeuralRecon/Test_data/OBL_L7/Test_imgs2_colmap_neural/sparse_align",
                                     "output/neural_recon/ngp_models",
                                     [1476])

    model = LModel19(data,
                     False,
                     [id_viz_face],
                     id_viz_edge,
                     [1, 0.01, 0],
                     "sample",
                     "imgs_log"
                     )

    state_dicts = torch.load("output/img_field_test/epoch=279-step=14280.ckpt")["state_dict"]
    state_dicts = {key[6:]:state_dicts[key] for key in state_dicts}
    model.load_state_dict(state_dicts)
    model.is_regress_normal=True
    model.eval()
    with torch.no_grad():
        for id_face in tqdm(range(len(model.batched_points_per_patch))):
            point_index = np.asarray(model.batched_points_per_patch[id_face]).reshape(-1,4)[:,0]
            ray_c = model.ray_c[point_index]
            distance = model.seg_distance[point_index] * model.seg_distance_normalizer
            points_c = ray_c * distance[:,None]

            is_detailed_log=False
            if is_detailed_log:
                id_points = torch.from_numpy(
                    np.concatenate(model.id_point_to_id_up_and_face[
                                       [item[0,0] in point_index for item in model.id_point_to_id_up_and_face]
                                   ])
                ).to(torch.long)
                id_start_point = id_points[:, 0]
                id_end_point = id_points[:,1]
                id_up = id_points[:,2:4]
                id_face = id_points[:,4:6]
                start_ray = model.ray_c[id_start_point]
                end_ray = model.ray_c[id_end_point]
                centroid_ray1 = model.center_ray_c[id_face[:, 0]]
                centroid_ray2 = model.center_ray_c[id_face[:, 1]]
                mask1 = id_face[:, 0] != -1
                mask2 = id_face[:, 1] != -1
                start_points = model.seg_distance[id_start_point][:, None] * model.seg_distance_normalizer * start_ray
                end_points = model.seg_distance[id_end_point][:, None] * model.seg_distance_normalizer * end_ray
                v_up = model.get_up_vector2(id_up, start_points, end_points)

                losses=[]
                loss_mask1s=[]
                loss_mask2s=[]
                nums=[]
                for dis in torch.arange(100,300,1):
                    model.seg_distance[point_index[0]] = dis / model.seg_distance_normalizer
                    loss, loss_mask1, loss_mask2, num = model.compute_similarity_wrapper(
                        start_ray, end_ray, model.seg_distance[id_start_point], model.seg_distance[id_end_point],
                        v_up, centroid_ray1, centroid_ray2, mask1, mask2
                    )
                    losses.append(loss.mean())
                    loss_mask1s.append(loss_mask1)
                    loss_mask2s.append(loss_mask2)
                    nums.append(num)


            line_thickness = 1
            point_thickness = 2
            point_radius = 1

            line_img1 = model.rgb1.copy()
            line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
            shape = line_img1.shape[:2][::-1]

            polygon_2d1 = (model.intrinsic1 @ points_c.T).T
            polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
            polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon_2d1], True,
                          color=(0, 0, 255), thickness=line_thickness)
            for i_line in range(polygon_2d1.shape[0]):
                cv2.circle(line_img1, polygon_2d1[i_line], radius=point_radius, color=(0, 255, 0), thickness=point_thickness)
                cv2.circle(line_img1, polygon_2d1[i_line], radius=point_radius, color=(0, 255, 0), thickness=point_thickness)

            min1 = np.asarray([
                np.clip(polygon_2d1[:,0].min() - 50, 0, shape[0]),
                np.clip(polygon_2d1[:,1].min() - 50, 0, shape[1]),
            ])
            max1 = np.asarray([
                np.clip(polygon_2d1[:, 0].max() + 50, 0, shape[0]),
                np.clip(polygon_2d1[:, 1].max() + 50, 0, shape[1]),
            ])

            img1 = line_img1[min1[1]:max1[1],min1[0]:max1[0]]

            # Image 2
            line_img2 = model.rgb2.copy()
            line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
            shape = line_img2.shape[:2][::-1]

            polygon_2d2 = (model.transformation @ to_homogeneous_tensor(points_c).T).T
            polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
            polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img2, [polygon_2d2], True,
                          color=(0, 0, 255), thickness=line_thickness)
            for i_line in range(polygon_2d2.shape[0]):
                cv2.circle(line_img2, polygon_2d2[i_line], radius=point_radius, color=(0, 255, 0), thickness=point_thickness)
                cv2.circle(line_img2, polygon_2d2[i_line], radius=point_radius, color=(0, 255, 0), thickness=point_thickness)

            min2 = np.asarray([
                np.clip(polygon_2d2[:, 0].min() - 50, 0, shape[0]),
                np.clip(polygon_2d2[:, 1].min() - 50, 0, shape[1]),
            ])
            max2 = np.asarray([
                np.clip(polygon_2d2[:, 0].max() + 50, 0, shape[0]),
                np.clip(polygon_2d2[:, 1].max() + 50, 0, shape[1]),
            ])

            img2 = line_img2[min2[1]:max2[1], min2[0]:max2[0]]

            cv2.imwrite(os.path.join("outputs/eval", "{:05d}.jpg".format(id_face)),
                        pad_imgs(img1,img2),)

            id_vertices = 0
            with open("outputs/eval/1.obj", "w") as f:
                points_w = ((torch.linalg.inv(model.extrinsic1) @ to_homogeneous_tensor(points_c).T).T)[:,:3]

                for i_line in range(polygon_2d1.shape[0]):
                    a = (i_line+1)%polygon_2d1.shape[0]
                    f.write("v {} {} {}\n".format(points_w[i_line,0], points_w[i_line,1], points_w[i_line,2]))
                    f.write("v {} {} {}\n".format(points_w[a,0], points_w[a,1], points_w[a,2]))
                    f.write("l {} {}\n".format(id_vertices+1, id_vertices+2))
                    id_vertices+=2

        pass
