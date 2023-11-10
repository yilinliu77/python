import os

import open3d as o3d

root = r"G:\Dataset\GSP\test_data2\mesh"

def viz(item):
    print(item)
    filename = os.path.join(root, item)
    # filename = os.path.join(root, item,
    #                         [file for file in os.listdir(os.path.join(root, item)) if file.split(".")[0]=="norm_mesh"][0])
    mesh = o3d.io.read_triangle_mesh(filename)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

if __name__ == '__main__':
    files = os.listdir(root)
    for item in files:
        viz(item)