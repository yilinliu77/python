from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import os


class Anchors(nn.Module):
    """ Anchor modules for multi-level dense output.

    """

    def __init__(self,
                 pyramid_levels: List[int],
                 sizes: List[float],
                 v_strides: List[float],
                 ratios: List[float],
                 scales: List[float],
                 preprocessed_path: str = ""
                 ):
        super(Anchors, self).__init__()
        assert len(pyramid_levels) == len(sizes) == len(v_strides)
        self.pyramid_levels = pyramid_levels
        self.sizes = sizes
        self.strides = v_strides
        self.ratios = ratios
        self.scales = scales
        self.preprocessed_path = preprocessed_path

        if self.preprocessed_path != "":
            self.anchors_mean_std = torch.from_numpy(np.load(self.preprocessed_path))

        self.anchors = None

    def forward(self, v_shape: np.ndarray):
        shape = np.asarray(v_shape)
        if self.anchors is None:
            image_shapes = [(shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

            # compute anchors over all pyramid levels
            all_anchors = np.zeros((0, 4)).astype(np.float32)

            for idx, p in enumerate(self.pyramid_levels):
                anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                shifted_anchors = shift(image_shapes[idx], p, self.strides[idx], anchors)
                all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
            self.anchors = torch.from_numpy(all_anchors)

            # self.anchors_image_x_center = self.anchors[:, 0:4:2].mean(dim=1)  # [N]
            # self.anchors_image_y_center = self.anchors[:, 1:4:2].mean(dim=1)  # [N]

        return self.anchors

    @property
    def num_anchors(self):
        return len(self.pyramid_levels) * len(self.ratios) * len(self.scales)

    @property
    def num_anchor_per_scale(self):
        return len(self.ratios) * len(self.scales)


def generate_anchors(base_size=None, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if base_size is None:
        base_size = 16

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(v_shape, v_pyramid, v_stride, v_anchors):
    shape_scale = 2 ** v_pyramid
    shift_x = (np.arange(0, v_shape[1], v_stride) + 0.5) * shape_scale
    shift_y = (np.arange(0, v_shape[0], v_stride) + 0.5) * shape_scale

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)re
    # reshape to (K*A, 4) shifted anchors
    A = v_anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (v_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


if __name__ == '__main__':
    anchor_generator = Anchors(
        pyramid_levels=[4, 5],
        sizes=[8, 4],
        v_strides=[1, 1],
        ratios=[.25, .5, 1, 2., 4.],
        scales=np.array([2 ** (i / 2.0) for i in range(8)]).tolist()
    )

    anchor1 = anchor_generator(np.zeros((800, 800)).shape)
    anchor2 = anchor_generator(np.zeros((360, 1280)).shape)

    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    start_index = 0
    for i_scale in [3, 4, 5]:
        shape = (img.shape[1] // (2 ** i_scale), img.shape[0] // (2 ** i_scale))

        for i in range(1000):
            viz_img = img.copy()
            anchor = anchor1[int(start_index +
                                 anchor_generator.num_anchor_per_scale * shape[0] * shape[1] / 4 / 4 / 2) + i]
            pts = list(map(int, anchor))
            viz_img = cv2.rectangle(viz_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 5)
            cv2.imshow("", viz_img)
            cv2.waitKey()
        start_index += anchor_generator.num_anchor_per_scale * shape[0] * shape[1] / 4 / 4

    cv2.destroyAllWindows()
    pass
