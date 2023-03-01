import sys

import numpy as np

from shared.common_utils import to_homogeneous

sys.path.append("thirdparty/sdf_computer/build/")


def normalize_intrinsic_and_extrinsic(v_bounds_center, v_bounds_size, v_img_size, v_intrinsic, v_extrinsics):
    assert v_intrinsic.shape == (3, 3)
    assert v_extrinsics.shape[1:] == (3, 4)

    model_matrix = np.zeros((4, 4), dtype=np.float32)
    model_matrix[0, 0] = v_bounds_size
    model_matrix[1, 1] = v_bounds_size
    model_matrix[2, 2] = v_bounds_size
    model_matrix[0, 3] = v_bounds_center[0] - v_bounds_size / 2
    model_matrix[1, 3] = v_bounds_center[1] - v_bounds_size / 2
    model_matrix[2, 3] = v_bounds_center[2] - v_bounds_size / 2
    model_matrix[3, 3] = 1

    intrinsic = to_homogeneous(np.copy(v_intrinsic))
    intrinsic[0, 0] /= v_img_size[0]
    intrinsic[0, 2] /= v_img_size[0]
    intrinsic[1, 1] /= v_img_size[1]
    intrinsic[1, 2] /= v_img_size[1]

    extrinsics = [to_homogeneous(item) @ model_matrix for item in v_extrinsics]

    projections = [to_homogeneous(intrinsic) @ item for item in extrinsics]
    positions = [(np.linalg.inv(to_homogeneous(item))[:3, 3] - v_bounds_center) / v_bounds_size + 0.5 for item in v_extrinsics]

    return intrinsic, extrinsics, positions, projections


