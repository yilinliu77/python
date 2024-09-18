import numpy as np
import open3d as o3d
from scipy.interpolate import BSpline


def check_points_on_plane(sample_points, abcd, tolerance=1e-6):
    a, b, c, d = abcd.flatten()  # 确保abcd是1D数组
    # 计算平面方程的左侧
    plane_eq = np.dot(sample_points, np.array([a, b, c])) + d
    # 检查所有点是否满足平面方程（在给定容差内）
    return np.all(np.abs(plane_eq) <= tolerance)


def sample_points_on_line(location, direction, vert_parameters, num_samples=20):
    norm_direction = direction / np.linalg.norm(direction)
    min_t, max_t = np.min(vert_parameters), np.max(vert_parameters)
    t_intervals = np.linspace(min_t, max_t, num_samples)
    sample_points = location[None, :].repeat(num_samples, 0) + norm_direction * t_intervals[:, None].repeat(3, 1)
    return sample_points


def sample_points_on_bspline(knots, poles, degree, rational, weights, vert_parameters, num_samples=20):
    t_min, t_max = np.min(vert_parameters), np.max(vert_parameters)
    t_interval = np.linspace(t_min, t_max, num_samples)

    bspline_curve = BSpline(knots, poles, degree)

    sampled_points = bspline_curve(t_interval)

    if rational:
        homogenous_points = np.hstack((sampled_points, np.ones((len(sampled_points), 1))))
        weighted_points = homogenous_points * weights[:, np.newaxis]
        sampled_points = weighted_points[:, :-1] / weighted_points[:, -1][:, np.newaxis]

    return sampled_points


def sample_points_on_circle(location, xyz_axis, radius, num_samples=20):
    x_axis, y_axis, z_axis = xyz_axis[0], xyz_axis[1], xyz_axis[2]
    theta_interval = np.linspace(0, 2 * np.pi, num_samples)
    x_component = radius * np.cos(theta_interval)[:, None] * x_axis
    y_component = radius * np.sin(theta_interval)[:, None] * y_axis

    sample_points = location + x_component + y_component

    return sample_points


def sample_points_on_plane(location, abcd, xyz_axis, vert_parameters, num_samples=20):
    a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]
    x_axis, y_axis, z_axis = xyz_axis[0], xyz_axis[1], xyz_axis[2]
    x_min, x_max = np.min(vert_parameters[:, 0]), np.max(vert_parameters[:, 0])
    y_min, y_max = np.min(vert_parameters[:, 1]), np.max(vert_parameters[:, 1])
    x_interval = np.linspace(x_min, x_max, num_samples)
    y_interval = np.linspace(y_min, y_max, num_samples)

    x_grid, y_grid = np.meshgrid(x_interval, y_interval)
    x_component = x_grid[..., np.newaxis] * x_axis
    y_component = y_grid[..., np.newaxis] * y_axis

    sample_points = location + x_component + y_component

    sample_points = sample_points.reshape(-1, 3)

    # if not check_points_on_plane(sample_points.copy(), abcd):
    #     raise ValueError("Sample points are not on the plane.")

    return sample_points


def get_plane_points(location, abcd, xyz_axis, vert_parameters):
    a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]
    x_axis, y_axis, z_axis = xyz_axis[0], xyz_axis[1], xyz_axis[2]

    points = vert_parameters[:, 0][:, None] * x_axis + vert_parameters[:, 1][:, None] * y_axis + location

    if not check_points_on_plane(points.copy(), abcd):
        raise ValueError("Sample points are not on the plane.")

    return points
