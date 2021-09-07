import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2 as cv
import math
from scipy.spatial.transform import Rotation as R

"""
Install the corresponding library
"""

"""
Given a set of 2D bounding box on an image, try to visualize them using opencv. The line should be drew in green and 
bold way. Also try to visualize both the gray and color image with bounding box using matplotlib.
params:
    - v_box: (N * 4) (N is the number of boxes. Each row is ordered as x1,y1,x2,y2 to represent a bounding box on the image)
Note:
    - The bounding box may be overrun the range of the image, try make the code robust
    - Links might be help:
        - https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
        - https://matplotlib.org/stable/tutorials/index.html
"""
def limit_range(v_value,v_min,v_max):
    if v_value<v_min:
        return v_min
    if v_value>v_max:
        return v_max
    return v_value

def draw_box(v_box):
    img = cv.imread(cv.samples.findFile("test.png"))
    
    sp = img.shape
    height = sp[0]
    width = sp[1]
    for box in v_box:
        box[0] = limit_range(box[0],0,width)
        box[1] = limit_range(box[1],0,height)
        box[2] = limit_range(box[2],0,width)
        box[3] = limit_range(box[3],0,height)
        
    for box in v_box:
        cv.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),3)
    cv.imshow("test", img)
    cv.waitKey()

    plt.title("test")
    #for row in img:
    #    for i in row:
    #        gray = i[2]*0.299 + i[1]*0.587 + i[0]*0.114
    #        i[0]=gray
    #        i[1]=gray
    #        i[2]=gray
    gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    #if just use once cvtColor, it will show a green picture.
    plt.imshow(gray)

import os
from visualDet3D.data.kitti.kittidata import KittiLabel

import torch
import torch.nn as nn

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    theta = alpha + np.arctan2(x + offset, z)
    return theta


class BBox3dProjector(nn.Module):
    """
        forward methods
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
    """
    def __init__(self):
        super(BBox3dProjector, self).__init__()
        self.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]]
        ).float()  )# 8, 3

    def forward(self, bbox_3d, tensor_p2):
        # """
        #     input:
        #         unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
        #         tensor_p2: tensor of [3, 4]
        #     output:
        #         [N, 3]
        #         [N, 8, 3] with corner point in camera frame # 8 is determined by the shape of self.corner_matrix
        #         [N, 8, 3] with corner point in image frame
        #         [N, ] thetas
        # """

        abs_corners = []
        homo_coords = []
        thetas = []

        r = R.from_rotvec([63 / 180 * np.pi, 0, 0])

        for i in range(len(bbox_3d)):
            # box: [x_min, y_min, x_max, y_max]
            # dim: [h, w, l]
            # center: [x, y, z]

            dim, center, alpha = bbox_3d[i, 3:6], bbox_3d[i, :3], bbox_3d[i, 6]

            rot_y = alpha2theta_3d(alpha, center[0], center[2], tensor_p2)
            thetas.append(rot_y)

            #print(alpha, dim, center)

            abs_corner = []
            homo_coord = []

            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [0, 1]:
                        point = np.copy(center) # center: [x, y, z], dim: [h, w, l]
                        point[0] = center[0] + i * dim[1] / 2 * np.cos(-rot_y - np.pi / 2) + (j * i) * dim[
                            2] / 2 * np.cos(-rot_y)
                        point[2] = center[2] + i * dim[1] / 2 * np.sin(-rot_y - np.pi / 2) + (j * i) * dim[
                            2] / 2 * np.sin(-rot_y)
                        point[1] = center[1] - k * dim[0]

                        point = r.apply(point)
                        point[2] = max(1., point[2])

                        abs_corner.append(point)

                        point = np.append(point, 1)

                        #print(".....")
                        #print(point)

                        point = np.dot(tensor_p2, point)
                        point = point / point[2]
                        point = point.astype(np.int16)
                        homo_coord.append(point)

            abs_corners.append(abs_corner)
            homo_coords.append(homo_coord)

        return torch.tensor(abs_corners), torch.tensor(homo_coords), thetas


class BackProjection(nn.Module):
    """
        forward method:
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
    """
    
    def forward(self, bbox3d, p2):
        """
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
        """
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]
        
        z3d = bbox3d[:, 2:3]  # [N, 1]
        x3d = (bbox3d[:, 0:1] * z3d - cx * z3d - tx) / fx  # [N, 1]
        y3d = (bbox3d[:, 1:2] * z3d - cy * z3d - ty) / fy  # [N, 1]
        
        r = R.from_rotvec([-63 / 180 * np.pi, 0, 0])

        # point = np.array([x3d, y3d, z3d]).permute(2,1)
        
        point = np.hstack((x3d, y3d, z3d))
        
        point = np.array(r.apply(point))

        # x3d, y3d, z3d = point
        
        ret = np.concatenate([point, bbox3d[:, 3:]], axis=1)
        
        return ret


def visual_3d(img, gt_3d, pred_3d, p2):
    
    back_project = BackProjection()
    
    gt_3d = back_project(gt_3d, p2)
    pred_3d = back_project(pred_3d, p2)
    
    # gt
    gt_centers = gt_3d[:, :3]
    gt_dims = gt_3d[:, 3:6]
    gt_alphas = gt_3d[:, 6]
    
    # pred
    pred_centers = pred_3d[:, :3]
    pred_dims = pred_3d[:, 3:6]
    pred_alphas = pred_3d[:, 6]
    
    for i in range(len(gt_centers)):

        # box: [x_min, y_min, x_max, y_max]
        # dim: [h, w, l]
        # center: [x, y, z]

        r = R.from_rotvec([63 / 180 * np.pi, 0, 0])

        dim, center, alpha = gt_dims[i], gt_centers[i], gt_alphas[i]

        rot_y = float(alpha) ## + np.arctan(center[0] / center[2])  # float(line[14])

        box_3d = []

        for i in [1, -1]:
            for j in [1, -1]:
                for k in [0, 1]:
                    point = np.copy(center) # center: [x, y, z], dim: [h, w, l]
                    point[0] = center[0] + i * dim[1] / 2 * np.cos(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dim[1] / 2 * np.sin(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.sin(-rot_y)
                    point[1] = center[1] - k * dim[0]

                    point = r.apply(point)
                    point[2] = max(1., point[2])

                    point = np.append(point, 1)

                    point = np.dot(p2, point)
                    point = point[:2] / point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        for i in range(4):
            point_1_ = box_3d[2 * i]
            point_2_ = box_3d[2 * i + 1]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)
    
    for i in range(len(gt_centers)):

        # box: [x_min, y_min, x_max, y_max]
        # dim: [h, w, l]
        # center: [x, y, z]

        r = R.from_rotvec([63 / 180 * np.pi, 0, 0])

        dim, center, alpha = pred_dims[i], pred_centers[i], pred_alphas[i]

        rot_y = float(alpha) ## + np.arctan(center[0] / center[2])  # float(line[14])

        box_3d = []

        for i in [1, -1]:
            for j in [1, -1]:
                for k in [0, 1]:
                    point = np.copy(center) # center: [x, y, z], dim: [h, w, l]
                    point[0] = center[0] + i * dim[1] / 2 * np.cos(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dim[1] / 2 * np.sin(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.sin(-rot_y)
                    point[1] = center[1] - k * dim[0]

                    point = r.apply(point)
                    point[2] = max(1., point[2])

                    point = np.append(point, 1)

                    point = np.dot(p2, point)
                    point = point[:2] / point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        for i in range(4):
            point_1_ = box_3d[2 * i]
            point_2_ = box_3d[2 * i + 1]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)
    
    return img
    
    

def draw_3D_box(img, corners, color = (255, 255, 0)):
    """
        draw 3D box in image with OpenCV,
        the order of the corners should be the same with BBox3dProjector
    """
    # points = np.array(corners[0:2], dtype=np.int32) #[2, 8]
    # points = [tuple(points[:,i]) for i in range(8)]
    # for i in range(1, 5):
    #     cv.line(img, points[i], points[(i%4+1)], color, 2)
    #     cv.line(img, points[(i + 4)%8], points[((i)%4 + 5)%8], color, 2)
    # cv.line(img, points[2], points[7], color)
    # cv.line(img, points[3], points[6], color)
    # cv.line(img, points[4], points[5],color)
    # cv.line(img, points[0], points[1], color)

    for k in range(corners.shape[0]):

        for i in range(4):
            point_1_ = corners[k][2 * i]
            point_2_ = corners[k][2 * i + 1]
            cv.line(img, (point_1_[0].item(), point_1_[1].item()), (point_2_[0].item(), point_2_[1].item()), color, 3)

        for i in range(8):
            point_1_ = corners[k][i]
            point_2_ = corners[k][(i + 2) % 8]
            cv.line(img, (point_1_[0].item(), point_1_[1].item()), (point_2_[0].item(), point_2_[1].item()), color, 3)

    return img

def draw_3d_bbox_ga(image_dir, calib_dir, label_dir, name, calib = None):

    image_path = os.path.join(image_dir, name)
    calib_path = os.path.join(calib_dir, name.replace('png', 'txt'))
    label_path = os.path.join(label_dir, name.replace('png', 'txt'))

    if calib is None:
        # read calibration data
        for line in open(calib_path):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))
    else:
        cam_to_img = calib

    img = cv.imread(image_path)

    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()

    kitti_label.data = [item for item in kitti_label.data if item.type == "Car"]

    projector = BBox3dProjector()

    bbox_3d = np.array([[obj.x, obj.y, obj.z, obj.h, obj.w, obj.l, obj.alpha] for obj in kitti_label.data])

    abs_corners, homo_corners, theta = projector(torch.tensor(bbox_3d), torch.tensor(cam_to_img))
    ##corners = homo_corners.permute(2,1,0)[:,:,0]

    out_img = draw_3D_box(img, homo_corners)

    cv.imshow("test", out_img)
    plt.show()

    cv.waitKey()

def draw_center_pt(image_dir, calib_dir, label_dir, name, calib = None):

    image_path = os.path.join(image_dir, name)
    calib_path = os.path.join(calib_dir, name.replace('png', 'txt'))
    label_path = os.path.join(label_dir, name.replace('png', 'txt'))

    if calib is None:
        # read calibration data
        for line in open(calib_path):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))
    else:
        cam_to_img = calib

    img = cv.imread(image_path)

    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()

    kitti_label.data = [item for item in kitti_label.data if item.type == "Car"]

    centers = np.array([[obj.x, obj.y, obj.z] for obj in kitti_label.data])

    r = R.from_rotvec([63/180 * np.pi, 0, 0])

    rot_centers = r.apply(centers)

    rot_centers = np.append(rot_centers, np.ones([len(rot_centers), 1]), axis=1)

    point = np.dot(cam_to_img, rot_centers.T)

    point = point[:2] / point[2]

    out_img = img

    out_pts = []

    origin = np.dot(cam_to_img, np.array([0,0,10,1]))

    origin = origin[:2] / origin[2]

    # draw origin point
    out_img = cv.circle(out_img, (int(origin[0]), int(origin[1])), radius=0, color=(255, 0, 0), thickness=10)

    for i in range(point.shape[1]):
        out_img = cv.circle(out_img, (int(point[0][i]), int(point[1][i])), radius=0, color=(0, 0, 255), thickness=10)
        out_pts.append([int(point[0][i]), int(point[1][i])])

    cv.imshow("test", out_img)
    plt.show()

    return point, out_pts, origin

def draw_2d_bbox(image_dir, label_dir, name):

    #image_path = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\testing\\image_2"
    #output_path = "C:\\Users\\zihan\\Desktop\\python\\src\\3d_detection\\temp\\kitti_test2\\output"
    #image_path = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\training\\image_2"

    image_path = os.path.join(image_dir, name)
    label_path = os.path.join(label_dir, name.replace('png', 'txt'))

    img = cv.imread(image_path)
    print(name)
    print(img.shape)

    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()

    kitti_label.data = [item for item in kitti_label.data if item.type == "Car"]

    v_box = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in kitti_label.data])

    sp = img.shape
    height = sp[0]
    width = sp[1]

    for box in v_box:
        box[0] = limit_range(box[0], 0, width)
        box[1] = limit_range(box[1], 0, height)
        box[2] = limit_range(box[2], 0, width)
        box[3] = limit_range(box[3], 0, height)

    for box in v_box:
        print(box)
        cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
    cv.imshow("test", img)
    cv.waitKey()

    plt.title("test")
    # for row in img:
    #    for i in row:
    #        gray = i[2]*0.299 + i[1]*0.587 + i[0]*0.114
    #        i[0]=gray
    #        i[1]=gray
    #        i[2]=gray
    gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # if just use once cvtColor, it will show a green picture.
    plt.imshow(gray)
    for box in v_box:
        plt.vlines(box[0], box[1], box[3], "g")
        plt.vlines(box[2], box[1], box[3], "g")
        plt.hlines(box[1], box[0], box[2], "g")
        plt.hlines(box[3], box[0], box[2], "g")
    plt.show()

    pass

def draw_3d_bbox(image_dir, calib_dir, label_dir, name, calib=None):

    image_path = os.path.join(image_dir, name)
    calib_path = os.path.join(calib_dir, name.replace('png', 'txt'))
    label_path = os.path.join(label_dir, name.replace('png', 'txt'))

    if calib is None:
        # read calibration data
        for line in open(calib_path):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))
    else:
        cam_to_img = calib

    img = cv.imread(image_path)

    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()

    kitti_label.data = [item for item in kitti_label.data if item.type == "Car"]

    v_box = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in kitti_label.data])
    dims = np.array([[obj.h, obj.w, obj.l] for obj in kitti_label.data])
    centers = np.array([[obj.x, obj.y, obj.z] for obj in kitti_label.data])
    alphas = np.array([obj.ry for obj in kitti_label.data])

    centers = np.append(centers, np.ones([len(centers), 1]), axis=1)
    ct_points = np.dot(cam_to_img, centers.T)
    ct_points = ct_points[:2] / ct_points[2]
    ct_points = ct_points.T

    sp = img.shape
    height = sp[0]
    width = sp[1]

    for box in v_box:
        box[0] = limit_range(box[0], 0, width)
        box[1] = limit_range(box[1], 0, height)
        box[2] = limit_range(box[2], 0, width)
        box[3] = limit_range(box[3], 0, height)

    for i in range(len(v_box)):

        # box: [x_min, y_min, x_max, y_max]
        # dim: [h, w, l]
        # center: [x, y, z]

        box, dim, center, alpha = v_box[i], dims[i], centers[i], alphas[i]

        ct_point = ct_points[i]

        print(alpha, box, dim, center)

        # draw bbox 2d
        #cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

        # if np.abs(float(alpha)) < 0.01:
        #     continue
        print
        alpha, center

        rot_y = float(alpha) ## + np.arctan(center[0] / center[2])  # float(line[14])

        box_3d = []

        for i in [1, -1]:
            for j in [1, -1]:
                for k in [0, 1]:
                    point = np.copy(center) # center: [x, y, z], dim: [h, w, l]
                    point[0] = center[0] + i * dim[1] / 2 * np.cos(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dim[1] / 2 * np.sin(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.sin(-rot_y)
                    point[1] = center[1] - k * dim[0]

                    point = np.dot(cam_to_img, point)
                    point = point[:2] / point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        print("ct_point: ")
        print(ct_point[:2])
        img = cv.circle(img, (int(ct_point[0]), int(ct_point[1])), radius=0, color=(0, 255, 0), thickness=10)

        for i in range(len(box_3d)):
            print(box_3d[i])
            #img = cv.circle(img, (int(box_3d[i][0]), int(box_3d[i][1])), radius=0, color=(0, 0, 255), thickness=10)

        for i in range(4):
            point_1_ = box_3d[2 * i]
            point_2_ = box_3d[2 * i + 1]
            #cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            #cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)


    cv.imshow("test", img)
    plt.show()

    cv.waitKey()

def draw_3d_bbox2(image_dir, calib_dir, label_dir, name, calib=None):

    image_path = os.path.join(image_dir, name)
    calib_path = os.path.join(calib_dir, name.replace('png', 'txt'))
    label_path = os.path.join(label_dir, name.replace('png', 'txt'))

    if calib is None:
        # read calibration data
        for line in open(calib_path):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))
    else:
        cam_to_img = calib

    img = cv.imread(image_path)

    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()

    kitti_label.data = [item for item in kitti_label.data if item.type == "Car"]

    v_box = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in kitti_label.data])
    dims = np.array([[obj.h, obj.w, obj.l] for obj in kitti_label.data])
    centers = np.array([[obj.x, obj.y, obj.z] for obj in kitti_label.data])
    alphas = np.array([obj.ry for obj in kitti_label.data])

    r = R.from_rotvec([63/180 * np.pi, 0, 0])

    rot_centers = r.apply(centers)

    rot_centers = np.append(rot_centers, np.ones([len(rot_centers), 1]), axis=1)
    ct_points = np.dot(cam_to_img, rot_centers.T)
    ct_points = ct_points[:2] / ct_points[2]
    ct_points = ct_points.T

    sp = img.shape
    height = sp[0]
    width = sp[1]

    for box in v_box:
        box[0] = limit_range(box[0], 0, width)
        box[1] = limit_range(box[1], 0, height)
        box[2] = limit_range(box[2], 0, width)
        box[3] = limit_range(box[3], 0, height)

    for i in range(len(v_box)):

        # box: [x_min, y_min, x_max, y_max]
        # dim: [h, w, l]
        # center: [x, y, z]

        box, dim, center, alpha = v_box[i], dims[i], centers[i], alphas[i]

        ct_point = ct_points[i]

        print(alpha, box, dim, center)

        # draw bbox 2d
        cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

        # if np.abs(float(alpha)) < 0.01:
        #     continue
        print
        alpha, center

        rot_y = float(alpha) ## + np.arctan(center[0] / center[2])  # float(line[14])

        box_3d = []

        for i in [1, -1]:
            for j in [1, -1]:
                for k in [0, 1]:
                    point = np.copy(center) # center: [x, y, z], dim: [h, w, l]
                    point[0] = center[0] + i * dim[1] / 2 * np.cos(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dim[1] / 2 * np.sin(-rot_y - np.pi / 2) + (j * i) * dim[
                        2] / 2 * np.sin(-rot_y)
                    point[1] = center[1] - k * dim[0]

                    point = r.apply(point)
                    point[2] = max(1., point[2])

                    point = np.append(point, 1)

                    print(".....")
                    print(point)

                    point = np.dot(cam_to_img, point)
                    point = point[:2] / point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        print("ct_point: ")
        print(ct_point[:2])
        img = cv.circle(img, (int(ct_point[0]), int(ct_point[1])), radius=0, color=(0, 255, 0), thickness=10)

        for i in range(len(box_3d)):
            print(box_3d[i])
            img = cv.circle(img, (int(box_3d[i][0]), int(box_3d[i][1])), radius=0, color=(0, 0, 255), thickness=10)

        for i in range(4):
            point_1_ = box_3d[2 * i]
            point_2_ = box_3d[2 * i + 1]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            cv.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)


    cv.imshow("test", img)
    plt.show()

    cv.waitKey()


if __name__ == '__main__':

    image_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\SZU\\training\\image_2"
    calib_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\SZU\\training\calib"
    label_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\SZU\\training\\label_2"
    name = "027047.png"

    # image_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\withAlpha\\training\\image_2"
    # calib_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\withAlpha\\training\\calib"
    # label_dir = "D:\\Unreal_model_split\\Shenzhen_dataset\\withAlpha\\training\\label_2"
    # name = "000000.png"
    #
    # image_dir = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\training\\image_2"
    # calib_dir = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\training\\calib"
    # label_dir = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\training\\label_2"
    # name = "000007.png"

    # image_dir = "C:\\Users\\zihan\\Desktop"
    # calib_dir = "C:\\Users\\zihan\\Desktop"
    # label_dir = "C:\\Users\\zihan\\Desktop"
    # name = "000000.png"

    calib = np.array([[400., 0., 400., 0.], [0., 400., 400., 0.], [0., 0., 1., 0.]])

    #draw_2d_bbox(image_dir, label_dir, name)
    #points, out_pts, ori = draw_center_pt(image_dir, calib_dir, label_dir, name, calib=calib)
    draw_3d_bbox_ga(image_dir, calib_dir, label_dir, name)

    # draw_3d_bbox(image_dir,s calib_dir, label_dir, name)
