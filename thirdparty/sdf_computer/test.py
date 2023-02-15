import sys,os,glob
import numpy as np

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import open3d as o3d

import pysdf # noqa

if __name__=="__main__":
    sdf_computer = pysdf.PYSDF_computer()

    mesh = o3d.io.read_triangle_mesh(
        "C:/DATASET/Test_imgs2_colmap_neural/sparse_align/detailed_l7_with_ground.obj")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    sdf_computer.setup_mesh(vertices[faces], True)
    
    cameras = []
    points = []
    cameras.append((53.011044, 38.780777, 29.532574))
    cameras.append((53.011044, 38.780777, 29.532574))
    points.append((53.011044, 38.780777, 9.632574))
    points.append((53.011044, 38.780777, 19.432574))
    points.append((53.011044, 38.780777, 19.532574))
    points.append((53.011044, 38.780777, 19.732574))
    cameras = (np.asarray(cameras) - sdf_computer.m_center) / sdf_computer.m_scale + 0.5
    points = (np.asarray(points) - sdf_computer.m_center) / sdf_computer.m_scale + 0.5
    visibility = sdf_computer.check_visibility(cameras, points)
    for item in visibility:
        print(item)

    sdf = sdf_computer.compute_sdf(int(1e6), int(1e6), int(1e6), False)

    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(sdf[:,:3])
    o3d.io.write_point_cloud("output/1.ply", pc)
    # pc.points=o3d.utility.Vector3dVector(sdf[(sdf[:,3])<0.001][:,:3])
    pc.points=o3d.utility.Vector3dVector(sdf[np.abs(sdf[:,3])<0.001][:,:3])
    o3d.io.write_point_cloud("output/2.ply", pc)

    pass