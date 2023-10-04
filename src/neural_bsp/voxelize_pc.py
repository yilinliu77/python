import open3d as o3d
import numpy as np

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh(
        "G:/Dataset/ABC/quadric_test/00999739/00999739_348b49253ff07a020fc5fe58_trimesh_000.obj")
    vertices = np.asarray(mesh.vertices)
    bb_min = np.min(vertices, axis=0)
    bb_max = np.max(vertices, axis=0)
    max_axis = np.max(bb_max - bb_min)
    vertices = (vertices - bb_min) / max_axis
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    pcd = mesh.sample_points_poisson_disk(10000)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,voxel_size=1/255, min_bound=np.zeros(3), max_bound=np.ones(3))
    fill_voxel = np.stack([item.grid_index for item in voxel_grid.get_voxels()], axis=0)

    voxel = np.zeros((256,256,256), dtype=bool)
    voxel[fill_voxel[:,0], fill_voxel[:,1], fill_voxel[:,2]] = True

    coords = np.transpose((np.mgrid[:256,:256,:256] / 255 * 2 - 1), (1,2,3,0))
    voxel_pc = coords[voxel]

    pc_mesh = o3d.geometry.PointCloud()
    pc_mesh.points = o3d.utility.Vector3dVector(voxel_pc)
    o3d.io.write_point_cloud("voxel_pc.ply", pc_mesh)

    np.savez("voxelized_point_cloud_256res_10000points",
             compressed_occupancies=np.packbits(voxel))

    pass