import open3d as o3d
import numpy as np

if __name__ == '__main__':
    flag = np.load(r"C:\repo\python\outputs\test_0808_26k_focal\00860128_flag.npy").reshape(
        (8,8,8,32,32,32)).transpose((0,3,1,4,2,5)).reshape((256,256,256)).astype(bool)
    coord = np.stack(np.meshgrid(
        np.arange(0, flag.shape[1]),
        np.arange(0, flag.shape[1]),
        np.arange(0, flag.shape[1]),
        indexing='ij'
    ),axis=3)/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord[flag])

    print(np.asarray(pcd.points).shape[0], " points")

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print(np.asarray(voxel_down_pcd.points).shape[0], " points after downsample")
    alpha = 0.1
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(voxel_down_pcd, alpha)
    mesh.compute_vertex_normals()

    # radii = [0.005, 0.01, 0.02, 0.04]
    # pcd.estimate_normals()
    # # pcd.orient_normals_consistent_tangent_plane(100)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii))

    o3d.visualization.draw_geometries([voxel_down_pcd])
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    exit()