import sys,os,glob
import numpy as np

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import open3d as o3d

import pysdf # noqa

if __name__=="__main__":
    sdf = pysdf.compute_sdf("/root/code/nglod/sdf-net/data/detailed_l7_with_ground.obj", int(1e6), int(1e6), int(1e6))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sdf[:, :3])
    o3d.io.write_point_cloud("/mnt/d/code/1.ply", pc)
    pc.points = o3d.utility.Vector3dVector(sdf[(sdf[:, 3]) < 0.001][:, :3])
    # pc.points=o3d.utility.Vector3dVector(sdf[np.abs(sdf[:,3])<0.001][:,:3])
    o3d.io.write_point_cloud("/mnt/d/code/2.ply", pc)

    mesh=o3d.io.read_triangle_mesh("/root/code/nglod/sdf-net/data/detailed_l7_with_ground.obj")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    sdf_computer = pysdf.SDF_computer(vertices[faces])
    sdf=sdf_computer.compute_sdf(int(1e6), int(1e6), int(1e6))

    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(sdf[:,:3])
    o3d.io.write_point_cloud("/mnt/d/code/1.ply", pc)
    # pc.points=o3d.utility.Vector3dVector(sdf[(sdf[:,3])<0.001][:,:3])
    pc.points=o3d.utility.Vector3dVector(sdf[np.abs(sdf[:,3])<0.001][:,:3])
    o3d.io.write_point_cloud("/mnt/d/code/2.ply", pc)

    pass