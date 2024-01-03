import numpy as np
import open3d as o3d
import os

root = r"G:\Dataset\GSP\test_data_small2\poisson"
output_root = r"G:\Dataset\GSP\test_data_small2\poisson_noise"

if __name__ == '__main__':
    os.makedirs(output_root, exist_ok=True)
    files = ([file for file in os.listdir(root) if file.endswith(".ply")])
    for file in files:
        pc = o3d.io.read_point_cloud(os.path.join(root, file))
        points = np.asarray(pc.points)
        box_min = np.min(points, axis=0)
        box_max = np.max(points, axis=0)
        diagonal = np.linalg.norm(box_max - box_min)
        noise_level = 0.01 * diagonal
        noise = (np.random.rand(points.shape[0], 3) - 0.5) * noise_level
        points += noise
        pc.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(output_root, file), pc)
    pass