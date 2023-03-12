import numpy as np
import numba as nb


def skew_matrix(a: np.ndarray):
    return np.array((
        (0, -a[2], a[1]),
        (a[2], 0, -a[0]),
        (-a[1], a[0], 0),
    ), dtype=a.dtype)


def extract_fundamental_from_projection(
        v_projection_matrix: np.ndarray,  # (4,4) transform from 1 to 2
        v_k1,  # (3,3)
        v_k2,  # (3,3)
):
    R = v_projection_matrix[:3, :3]
    t = v_projection_matrix[:3, 3]
    F = np.linalg.inv(np.transpose(v_k2)) @ skew_matrix(t) @ R @ np.linalg.inv(v_k1)
    return F