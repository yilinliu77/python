import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2 as cv
import math

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

def draw_box_result(name):

    image_path = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\testing\\image_2"
    output_path = "C:\\Users\\zihan\\Desktop\\python\\src\\3d_detection\\temp\\kitti_test2\\output"
    #image_path = "C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\\data\\kitti_obj\\training\\image_2"

    cv.samples.addSamplesDataSearchPath(image_path)
    img = cv.imread(cv.samples.findFile(name))
    print(name)
    print(img.shape)
    label_path = os.path.join(output_path, name[:6] + ".txt")
    kitti_label = KittiLabel(label_path=label_path)
    kitti_label.read_label_file()
    print(kitti_label.data)
    v_box = [[24.975143,171.052780,226.496078,294.367249]]

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
    for box in boxes:
        plt.vlines(box[0], box[1], box[3], "g")
        plt.vlines(box[2], box[1], box[3], "g")
        plt.hlines(box[1], box[0], box[2], "g")
        plt.hlines(box[3], box[0], box[2], "g")
    plt.show()

    pass

draw_box_result("000001.png")


if __name__ == '__main__':
    boxes = [
        [47, 246, 133, 468],
        [316, 142, 416, 471],
        [0, 400, 700, 600],
    ]
    draw_box(boxes)
