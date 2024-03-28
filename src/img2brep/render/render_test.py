import blenderproc as bproc
import argparse
import json
import os
import numpy as np
import bpy
from blenderproc.python.types.MeshObjectUtility import create_from_blender_mesh

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def add_point_light(energe=3000, location=[-5, -5, 5]):
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(energe)
    # light.set_type("SUN")
    light.set_location(location)


def Fibonacci_grid_sample(num, radius):
    # https://www.jianshu.com/p/8ffa122d2c15
    points = [[0, 0, 0] for _ in range(num)]
    phi = 0.618
    for n in range(num):
        z = (2 * n - 1) / num - 1
        x = np.sqrt(np.abs(1 - z * z)) * np.cos(2 * np.pi * n * phi)
        y = np.sqrt(np.abs(1 - z * z)) * np.sin(2 * np.pi * n * phi)
        points[n][0] = x * radius
        points[n][1] = y * radius
        points[n][2] = z * radius

    points = np.array(points)
    return points


def sphere_angle_sample(num, radius):
    points = []
    for azim in np.linspace(-180, 180, num):
        elev = 60
        razim = np.pi * azim / 180
        relev = np.pi * elev / 180

        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points

def sphere_angle_sample2(num, radius):
    points = []
    for azim in [-60,60,180]:
        elev = 60
        razim = np.pi * azim / 180
        relev = np.pi * elev / 180

        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points

def sphere_angle_sample_new(num, radius):
    num = int(np.math.sqrt(num))
    points = []
    for elev in np.linspace(-180, 180, num):
        for azim in np.linspace(-180, 180, num):
            razim = np.math.pi * azim / 180
            relev = np.math.pi * elev / 180

            center = [0, 0, 0]
            xp = center[0] + np.math.cos(razim) * np.math.cos(relev) * radius
            yp = center[1] + np.math.sin(razim) * np.math.cos(relev) * radius
            zp = center[2] + np.math.sin(relev) * radius
            points.append([xp, yp, zp])
    points = np.array(points)
    return points

def sphere_angle_sample_for_video(num, radius):
    points = []
    for indice in range(num):
        azim = indice
        elev = indice / 2
        if 90 < elev <= 180:
            elev = 180 - elev

        razim = np.pi * azim / 180
        relev = np.pi * elev / 180
        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points


parser = argparse.ArgumentParser()

parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('num', default=100, type=int, help="number of rendering")
parser.add_argument('split', default="train", type=str, help="train, val or test")

args = parser.parse_args()
scene_name = os.path.basename(args.scene)[:-4]

bproc.init()
objs = bproc.loader.load_obj(args.scene)
obj = objs[0]

mesh = obj.mesh_as_trimesh()
vertices = np.asarray(mesh.vertices)
min_coords = np.min(vertices, axis=0)
max_coords = np.max(vertices, axis=0)
center = (min_coords + max_coords) / 2
diag = np.linalg.norm(max_coords-min_coords)

obj.edit_mode()
blender_mesh = obj.mesh_as_bmesh()
for v in blender_mesh.verts:
    v.co.x = (v.co.x-center[0]) / diag * 2 * 0.8
    v.co.y = (v.co.y-center[1]) / diag * 2 * 0.8
    v.co.z = (v.co.z-center[2]) / diag * 2 * 0.8
obj.update_from_bmesh(blender_mesh)
obj.object_mode()
# obj = create_from_blender_mesh(blender_mesh)

mesh = obj.mesh_as_trimesh()
mesh.export("1.obj", "obj")
exit()

# Scale the 3D model
obj.set_location([0, 0, 0])
obj.set_rotation_euler([0, 0, 0])

bbox = obj.get_bound_box()
min_coords = bbox[0]
max_coords = bbox[-2]
center = (min_coords + max_coords) / 2
diag = np.linalg.norm(max_coords-min_coords)
scale = 1 / diag / 0.8 / 2
print("normalize scale:", scale)
origin = obj.get_origin()
print("min_coords:", min_coords)
print("max_coords:", max_coords)
print("center:", center)
print("origin:", origin)
obj.set_location(center - origin)
# obj.set_rotation_euler([0, 0, 0])
# obj.set_scale([scale, scale, scale])

mesh = obj.mesh_as_trimesh()
mesh.export("1.obj", "obj")
exit()


# add texture
texture_path = r"/home/duoteng/workspace/ABC/BlenderProc_NEF/images/material_manipulation_sample_texture.jpg"
texture_path1 = r"/home/duoteng/workspace/ABC/BlenderProc_NEF/images/material_manipulation_sample_texture1.jpg"
texture_path2 = r"/home/duoteng/workspace/ABC/BlenderProc_NEF/images/material_manipulation_sample_texture2.jpg"

# mat = objs[0].new_material("mat")
# image = bpy.data.images.load(filepath=texture_path)
# mat.set_principled_shader_value("Specular", image)
# mat.blender_obj.use_nodes = True
# node_tex = mat.blender_obj.node_tree.nodes.new("ShaderNodeTexEnvironment")
# node_tex.image = image
# links = mat.blender_obj.node_tree.links
# link = links.new(node_tex.outputs[0], mat.blender_obj.node_tree.nodes[0].inputs["Base Color"])


mat = objs[0].new_material("mat").blender_obj
mat.use_nodes = True

# Image texture
#image = bpy.data.images.load(filepath=r"C:/Users/whats/Downloads/sample_texture.jpg")
#node_tex = mat.node_tree.nodes.new("ShaderNodeTexEnvironment")
#node_tex.image = image

# Checker texture
node_tex = mat.node_tree.nodes.new('ShaderNodeTexChecker')
node_mixed = mat.node_tree.nodes.new('ShaderNodeMixShader')
node_ao = mat.node_tree.nodes.new('ShaderNodeAmbientOcclusion')
color_ramp = mat.node_tree.nodes.new("ShaderNodeValToRGB")

links = mat.node_tree.links
links.new(node_tex.outputs[0], mat.node_tree.nodes[0].inputs[0])
links.new(color_ramp.outputs[0], node_mixed.inputs[2])
links.new(node_ao.outputs[1], color_ramp.inputs[0])
links.new(node_mixed.outputs[0], mat.node_tree.nodes[1].inputs[0])
links.new(mat.node_tree.nodes[0].outputs[0], node_mixed.inputs[1])

node_mixed.inputs[0].default_value = 0.2
color_ramp.color_ramp.elements[0].position = (0.2)
color_ramp.color_ramp.elements[1].position = (1.0)
node_ao.inputs[1].default_value=0.1

# define a light and set its location and energy level
add_point_light(energe=200, location=[-1.5, 1.5, 1.5])
add_point_light(energe=200, location=[1.5, 1.5, 1.5])
add_point_light(energe=200, location=[-1.5, -1.5, 1.5])
add_point_light(energe=200, location=[1.5, -1.5, 1.5])
add_point_light(energe=200, location=[-1.5, 1.5, -1.5])
add_point_light(energe=200, location=[1.5, 1.5, -1.5])
add_point_light(energe=200, location=[-1.5, -1.5, -1.5])
add_point_light(energe=200, location=[1.5, -1.5, -1.5])

# add_point_light(energe=500, location=[-1, 3, 3] + poi)
# add_point_light(energe=300, location=[-1, -1, 3] + poi)
# add_point_light(energe=500, location=[0, 0, -3] + poi)
# add_point_light(energe=300, location=[4, 0, 3] + poi)

add_point_light(energe=10, location=poi)   # set a light in the center of the obj (calculate the center, [0, 0, 0])


# define the camera resolution
bproc.camera.set_resolution(800, 800)
fov_x, fov_y, angle_x = bproc.camera.get_fov()    # just to get angle_x. see python/camera/CameraUtility.py for details or changes

if args.split == "test":
    locations = sphere_angle_sample(num=args.num, radius=3)       # for generating test set, and views for videos
    locations += poi
elif args.split == "video":
    locations = sphere_angle_sample_for_video(num=args.num, radius=3)       # for generating test set, and views for videos
    locations += poi
    print("sample cameras for video:", len(locations))
else:
    # locations = Fibonacci_grid_sample(num=args.num, radius=3)       # for generating training set
    locations = sphere_angle_sample2(num=args.num, radius=3)
    locations += poi

# Sample several camera poses
for i in range(args.num):
    # # Sample random camera location above objects
    # location = np.random.uniform([-1, -1, 0], [1, 1, 2])

    if args.split == "val":
        location = locations[args.num - 1 - i]
    else:
        location = locations[i]     # this is normal, the "val" is not.

    # # Sample random camera location around the object
    # location = bproc.sampler.sphere(poi, radius=3, mode="SURFACE")
    # location = bproc.sampler.sphere([0, 0, 0], radius=3, mode="SURFACE")

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # print(rotation_matrix)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

bproc.renderer.set_output_format(enable_transparency=True)
# render the whole pipeline
data = bproc.renderer.render()

# Collect state of the camera at all frames
cam_states = []
for frame in range(bproc.utility.num_frames()):
    cam_states.append({
        "cam2world": bproc.camera.get_camera_pose(frame),
        "cam_K": bproc.camera.get_intrinsics_as_K_matrix()
    })
# Adds states to the data dict
data["cam_states"] = cam_states

# write the data to a .hdf5 container
output_split_dir = os.path.join(args.output_dir, args.split)
os.makedirs(output_split_dir, exist_ok=True)
bproc.writer.write_hdf5(output_split_dir, data)

#存储相机参数到json文件
camera_json = {"camera_angle_x": angle_x, "frames": []}
for i in range(len(cam_states)):
    filename = str(i) + "_" + "colors"
    Rt = cam_states[i]["cam2world"]
    R = Rt[:3, :3]
    T = Rt[:3, 3:4]
    world2cam = np.concatenate((np.concatenate((R.transpose(), -(R.transpose().dot(T))), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    K = cam_states[i]["cam_K"]
    camera = {"file_path": "./" + args.split + "/" + filename,
                  "rotation": 0,   # not use
                  "camera_intrinsics": K.tolist(),
                  "transform_matrix": Rt.tolist()}

    camera_json["frames"].append(camera)

with open(os.path.join(os.path.dirname(args.output_dir), "transforms_" + args.split + ".json"), "w") as f:
    json.dump(camera_json, f, indent=4)

# blenderproc run examples/datasets/abc_dataset/main.py  examples/datasets/abc_dataset/00000003/00000003_1ffb81a71e5b402e966b9341_trimesh_002.obj examples/datasets/abc_dataset/output 100 train
# blenderproc vis hdf5 examples/datasets/abc_dataset/output/*.hdf5 --save examples/datasets/abc_dataset/output_rgb/


