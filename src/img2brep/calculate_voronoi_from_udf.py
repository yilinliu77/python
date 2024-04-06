import h5py
import numpy as np

from shared.common_utils import export_point_cloud

if __name__ == '__main__':
    with h5py.File(r"G:/Projects/img2brep/data/single_image_diffusion/training.h5", "r") as f:
        udf = np.asarray(f["features"], dtype=np.float64)[0, ..., 0] / 65535 * 2.
        g = np.asarray(f["features"], dtype=np.float64)[0, ..., 1:3]
        flags = np.asarray(f["flags"][0])[..., None] > 0

    vertices = np.stack(np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij'), axis=-1)
    vertices = vertices / 255 * 2 - 1

    # First derivative
    dx,dy,dz = np.gradient(udf)
    d1 = np.stack([dx,dy,dz], axis=-1)
    norm1 = np.linalg.norm(d1, axis=-1) + 1e-18
    d1_normalized = d1 / norm1[..., None]

    # Surface points
    phi = g[...,0] / 65535. * 2 * np.pi
    theta = g[...,1] / 65535. * 2 * np.pi
    direction = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    surface_points = vertices - d1_normalized * udf[...,None]
    export_point_cloud("surface.ply", surface_points.reshape(-1, 3))

    # d1_normalized = direction

    # Find the orthogonal vector to the gradient
    d_orthogonal = np.cross(d1_normalized, np.array([1.1, 1.1, 1.1]))
    d_orthogonal_normalized = d_orthogonal / (np.linalg.norm(d_orthogonal, axis=-1)[..., None] + 1e-18)

    # second derivative
    dx2,dy2,dz2,_ = np.gradient(d1_normalized)
    H = np.stack([dx2,dy2,dz2], axis=-1)

    # Gaussian curvature
    determinant = np.linalg.det(H)
    determinant = np.abs(determinant)

    # second derivative along the orthogonal vector
    # dx2 * d_orthogonal_normalized[..., 0:1] + dy2 * d_orthogonal_normalized[..., 1:2] + dz2 * d_orthogonal_normalized[..., 2:3]
    d2 = (H @ d_orthogonal_normalized[..., None])[...,0]
    norm2 = np.linalg.norm(d2, axis=-1) + 1e-18
    d2_normalized = d2 / norm2[..., None]

    # third derivative along the orthogonal vector
    dx3,dy3,dz3,_ = np.gradient(d2_normalized)
    H3 = np.stack([dx3,dy3,dz3], axis=-1)
    d3 = (H3 @ d_orthogonal_normalized[..., None])[...,0]
    norm3 = np.linalg.norm(d3, axis=-1) + 1e-18

    norm_9 = np.quantile(norm3, 0.9)
    threshold = 0.2
    flags = norm3 > threshold
    flags[:2] = False
    flags[-2:] = False
    flags[:,2] = False
    flags[:,-2:] = False
    flags[:,:,2] = False
    flags[:,:,-2:] = False
    export_point_cloud("1.ply", vertices[flags])

    pass