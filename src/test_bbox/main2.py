import numpy as np
from matplotlib import pyplot as plt
import cv2

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
def draw_box(v_box):
    pass


if __name__ == '__main__':
    boxes = [
        [47, 246, 133, 468],
        [316, 142, 416, 471],
        [0, 400, 700, 600],
    ]
    draw_box()
