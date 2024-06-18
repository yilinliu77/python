import os
import cv2
import torch
import trimesh
import numpy as np

from shared.common_utils import to_homogeneous_tensor
from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.phase_no_grad_edge import read_graph

if __name__ == '__main__':
    bound_min = np.array((0,0,0))
    bound_max = np.array((1,1,1))
    img_database, points_3d = read_dataset(r"G:/Projects/NeuralRecon/data/00016422",
                                           [bound_min, bound_max]
                                           )
    img = cv2.imread("G:/Projects/NeuralRecon/data/00016422/imgs/24_colors.png")
    img2= cv2.imread("G:/Projects/NeuralRecon/data/00016422/imgs/14_colors.png")

    graph = read_graph(r"G:/Projects/NeuralRecon/data/00016422/wireframe/24_colors.obj",
               r"G:/Projects/NeuralRecon/data/00016422/imgs/24_colors.png",
               [800,800],
               device=torch.device("cuda"))
    # img[img==0]=255

    # Epipolar
    id_face=10

    ref_Image = img_database[24]
    intrinsic = torch.from_numpy(ref_Image.intrinsic).to(torch.float32)
    ref_extrinsic = torch.from_numpy(ref_Image.extrinsic).to(torch.float32)
    src_extrinsic = torch.from_numpy(img_database[14].extrinsic).to(torch.float32)

    shape = img.shape[:2][::-1]
    shape_tensor = torch.tensor([shape[0], shape[1], 1]).to(torch.float32)

    pos_2d = torch.from_numpy(np.asarray(
        [graph.nodes[item]["pos_2d"] for item in graph.graph["src_faces"][id_face]])).to(torch.float32)

    p = intrinsic.inverse() @ to_homogeneous_tensor(pos_2d).T
    p1 = p * 1
    p2 = p * 10
    p1 = ref_extrinsic.inverse() @ to_homogeneous_tensor(p1.T).T
    p1 = (intrinsic @ (src_extrinsic @ p1)[:3, :]).T
    p1 = p1 / p1[:, 2:3] * shape_tensor
    p2 = ref_extrinsic.inverse() @ to_homogeneous_tensor(p2.T).T
    p2 = (intrinsic @ (src_extrinsic @ p2)[:3, :]).T
    p2 = p2 / p2[:, 2:3] * shape_tensor

    p1, p2 = p1[:, :2], p2[:, :2]
    a = p2[:, 1] - p1[:, 1]
    b = -(p2[:, 0] - p1[:, 0])
    c = p2[:, 0] * p1[:, 1] - p1[:, 0] * p2[:, 1]
    lines = torch.stack((a, b, c), dim=1)


    def calculate_intersections(a, b, c, W, H):
        points = []
        # Left
        y_left = -c / b if b != 0 else None
        if y_left is not None and 0 <= y_left <= H:
            points.append((0, y_left))
        # Right
        y_right = -(a * W + c) / b if b != 0 else None
        if y_right is not None and 0 <= y_right <= H:
            points.append((W, y_right))
        # Down
        x_bottom = -c / a if a != 0 else None
        if x_bottom is not None and 0 <= x_bottom <= W:
            points.append((x_bottom, 0))
        # Up
        x_top = -(b * H + c) / a if a != 0 else None
        if x_top is not None and 0 <= x_top <= W:
            points.append((x_top, H))
        return points

    def draw_line(image, abc, color=(0, 255, 0), thickness=2):

        a, b, c = int(abc[0]), int(abc[1]), int(abc[2])
        # Img size
        height, width = image.shape[:2]
        intersections = calculate_intersections(a, b, c, width, height)
        if len(intersections) == 2:
            pt1 = (int(intersections[0][0]), int(intersections[0][1]))
            pt2 = (int(intersections[1][0]), int(intersections[1][1]))
            cv2.line(image, pt1, pt2, color, thickness)

        return image

    for i in range(pos_2d.shape[0]):
        # Img1
        pts11 = np.around(pos_2d[i].cpu().numpy()[:2] * shape).astype(np.int64)
        pts21 = np.around(pos_2d[(i+1)%pos_2d.shape[0]].cpu().numpy()[:2] * shape).astype(np.int64)

        img_ = img.copy()
        cv2.line(img_,
                 tuple(pts11),
                 tuple(pts21), (255, 0, 0), 2)
        cv2.circle(img_, tuple(pts11), 2, (0, 0, 255), 2)
        cv2.circle(img_, tuple(pts21), 2, (0, 255, 0), 2)


        # Img2
        pts21, pts22 = calculate_intersections(
            lines[i][0].item(), lines[i][1].item(), lines[i][2].item(), shape[0], shape[1])
        pts23, pts24 = calculate_intersections(
            lines[
                (i+1)%pos_2d.shape[0]][0].item(),
            lines[(i+1)%pos_2d.shape[0]][1].item(),
            lines[(i+1)%pos_2d.shape[0]][2].item(), shape[0], shape[1])

        img2_ = img2.copy()
        cv2.line(img2_,
                 (int(pts21[0]), int(pts21[1])),
                 (int(pts22[0]), int(pts22[1])), (0, 0, 255), 2)
        cv2.line(img2_,
                    (int(pts23[0]), int(pts23[1])),
                    (int(pts24[0]), int(pts24[1])), (0, 255, 0), 2)
        img_show = np.hstack((img_, img2_))
        cv2.imshow("1", img_show)
        cv2.waitKey()

        if pts21[0]!=800:
            t = pts21
            pts21 = pts22
            pts22 = t
        if pts23[0]!=800:
            t = pts23
            pts23 = pts24
            pts24 = t
        tl = pts22 if pts22[1] < pts24[1] else pts24
        tr = pts21 if pts21[1] < pts23[1] else pts23
        bl = pts22 if pts22[1] > pts24[1] else pts24
        br = pts21 if pts21[1] > pts23[1] else pts23

        img2_ = img2.copy()
        cv2.fillPoly(img2_,
                      [np.asarray([tl, tr, br, bl]).astype(np.int32)],
                      color=np.random.randint(0, 255, 3).tolist())
        cv2.addWeighted(img2_, 0.5, img2, 0.5, 0, img2_)

        img_show = np.hstack((img_, img2_))
        cv2.imshow("1", img_show)
        cv2.waitKey()

    # Visualize face 10
    img_ = img.copy()
    for i in range(pos_2d.shape[0]):
        # Img1
        pts11 = np.around(pos_2d[i].cpu().numpy()[:2] * shape).astype(np.int64)
        pts21 = np.around(pos_2d[(i+1)%pos_2d.shape[0]].cpu().numpy()[:2] * shape).astype(np.int64)

        cv2.line(img_, tuple(pts11), tuple(pts21), (0, 0, 255), 2)

    cv2.imshow("img", img_)
    cv2.waitKey()

    # Visualize face 14
    pos_2d_14 = torch.from_numpy(np.asarray(
        [graph.nodes[item]["pos_2d"] for item in graph.graph["src_faces"][14]])).to(torch.float32)
    for i in range(pos_2d_14.shape[0]):
        # Img1
        pts11 = np.around(pos_2d_14[i].cpu().numpy()[:2] * shape).astype(np.int64)
        pts21 = np.around(pos_2d_14[(i+1)%pos_2d_14.shape[0]].cpu().numpy()[:2] * shape).astype(np.int64)
        cv2.line(img_, tuple(pts11), tuple(pts21), (0, 255, 0), 2)

    cv2.imshow("img", img_)
    cv2.waitKey()

    preserved_face = [2,3,4,5,8,9,10,14,15,18,19,20,21,22]
    for idx,face in enumerate(graph.graph["src_faces"]):
        if idx not in preserved_face:
            continue
        edges = list(zip(face, face[1:] + [face[0]]))

        img2 = img.copy()
        cv2.fillPoly(img2,
                      [np.asarray([(graph.nodes[face[i]]["pos_2d"] * 800) for i in range(len(face))]).astype(np.int32)],
                      color=np.random.randint(0, 255, 3).tolist())

        cv2.addWeighted(img2, 0.5, img, 0.5, 0, img)

        for edge in edges:
            v0 = graph.nodes[edge[0]]["pos_2d"] * 800
            v1 = graph.nodes[edge[1]]["pos_2d"] * 800
            cv2.line(img, (int(v0[0]), int(v0[1])), (int(v1[0]), int(v1[1])), (0, 0, 255), 1)
        print(idx)
        cv2.imshow("img", img)
        cv2.waitKey()

