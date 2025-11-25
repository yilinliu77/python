import os
import random
import shutil
from pathlib import Path

import ray, trimesh
import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial_Silver, Graphic3d_TypeOfShadingModel, \
    Graphic3d_NameOfMaterial_Brass, Graphic3d_TypeOfReflection, Graphic3d_AspectLine3d
from OCC.Core.Graphic3d import *
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Dir
from OCC.Display.OCCViewer import OffscreenRenderer, Viewer3d
from OCC.Extend.DataExchange import read_step_file, write_step_file
import traceback, sys

from PIL import Image
from scipy.spatial.transform import Rotation

from shared.occ_utils import normalize_shape, get_triangulations, get_primitives, get_ordered_edges
from scipy.spatial.transform import Rotation

debug_id = None
# debug_id = "00000797"

# data_root = Path(r"/mnt/d/img2brep/data_step")
# npz_root = Path(r"/mnt/d/img2brep/abc_v2_npz")
# img_root = Path(r"/mnt/d/img2brep/abc_v2_png")
# data_split = r"src/brepnet/data/list/abc_total.txt"

data_root = Path(r"/mnt/d/img2brep/data_step")
npz_root = Path(r"/mnt/d/img2brep/deepcad_v6_npz")
img_root = Path(r"/mnt/d/img2brep/deepcad_v6_png")
data_split = r"src/brepnet/data/list/deepcad_7_30.txt"

# data_root = Path(r"E:\data\img2brep\train_diffusion_test_data\data_step")
# npz_root = Path(r"E:\data\img2brep\train_diffusion_test_data\abc_v2_npz")
# img_root = Path(r"E:\data\img2brep\train_diffusion_test_data\abc_v2_png")
# data_split = r"D:\WorkSpace\brepnet\src\brepnet\data\list\abc_total.txt"

exception_files = [
]

num_max_primitives = 100000
sample_resolution = 16


def check_dir(v_path):
    if os.path.exists(v_path):
        shutil.rmtree(v_path)
    os.makedirs(v_path)


def safe_check_dir(v_path):
    if not os.path.exists(v_path):
        os.makedirs(v_path)

class MyOffscreenRenderer(Viewer3d):
    """The offscreen renderer is inherited from Viewer3d.
    The DisplayShape method is overridden to export to image
    each time it is called.
    """

    def __init__(self, screen_size=(224, 224)):
        super().__init__()
        # create the renderer
        self.Create(display_glinfo=False)
        self.SetSize(screen_size[0], screen_size[1])
        self.SetModeShaded()
        self.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
        self.capture_number = 0


def fibonacci_sphere_samples(n):
    """使用斐波那契螺旋在球面上均匀采样点"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角度

    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y从1到-1
        radius = np.sqrt(1 - y * y)  # 在y处的半径

        theta = phi * i  # 黄金角度旋转

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

# @ray.remote(num_cpus=1)
def get_brep(v_root, output_root, v_folders):
    # v_folders = ["00001000"]
    single_loop_folder = []
    random.seed(0)
    for idx, v_folder in enumerate(v_folders):
        # for idx, v_folder in enumerate(tqdm(v_folders)):
        safe_check_dir(output_root / v_folder)

        try:
            # Load mesh and yml files
            all_files = os.listdir(v_root / v_folder)
            step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff][0]

            # Start to extract BREP
            shape = read_step_file(str(v_root / v_folder / step_file), verbosity=False)
            if shape.NbChildren() != 1:
                raise ValueError("Multiple components: {}; Jump over".format(v_folder))
            shape = normalize_shape(shape, 0.9)

            #v,f = get_triangulations(shape)
            #trimesh.Trimesh(vertices=v, faces=f).export(str(output_root / v_folder / "mesh.obj"))

            data_dict = {
            }

            (img_root / v_folder).mkdir(parents=True, exist_ok=True)

            # Single view
            display = MyOffscreenRenderer()
            mat = Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_Silver)
            mat.SetReflectionModeOff(Graphic3d_TypeOfReflection.Graphic3d_TOR_SPECULAR)
            display.DisplayShape(shape, material=mat, update=True)
            display.camera.SetProjectionType(1)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))

            svr_imgs = []
            display.camera.SetEyeAndCenter(gp_Pnt(2., 2., 2.), gp_Pnt(0., 0., 0.))
            display.camera.SetUp(gp_Dir(0,0,1))
            display.camera.SetAspect(1)
            display.camera.SetFOVy(45)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))

            init_pos = np.array((2, 2, 2))
            up_pos = np.array((2, 2, 3))

            x_steps = 4  # x轴8个角度
            y_steps = 4  # y轴8个角度
            z_steps = 4  # z轴4个角度（通常不需要太多z轴旋转）
            total_views = x_steps * y_steps * z_steps
            for i in range(total_views):
                angles = np.array([
                    i % x_steps,
                    i // x_steps % y_steps,
                    i // (x_steps * y_steps)
                ])
                matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
                view = (matrix @ init_pos.T).T
                up = (matrix @ up_pos.T).T

                # new_v = (matrix.T @ np.asarray(v).T).T
                # trimesh.Trimesh(vertices=new_v, faces=f).export(str(img_root / v_folder / f"mesh_{i}.ply"))

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))

                display.camera.SetProjectionType(0)
                filename = str(img_root / v_folder / f"shape_{i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                display.View.Dump(str(img_root / v_folder / f"view_.png"))
                svr_imgs.append(img1)

            x_steps = 4  # x轴8个角度
            y_steps = 4  # y轴8个角度
            z_steps = 4  # z轴4个角度（通常不需要太多z轴旋转）
            total_views = x_steps * y_steps * z_steps
            for i in range(total_views):
                angles = np.array([
                    i % x_steps,
                    i // x_steps % y_steps,
                    i // (x_steps * y_steps)
                ])
                matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
                view = (matrix @ init_pos.T).T
                up = (matrix @ up_pos.T).T

                # new_v = (matrix.T @ np.asarray(v).T).T
                # trimesh.Trimesh(vertices=new_v, faces=f).export(str(img_root / v_folder / f"mesh_{i}.ply"))

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))

                display.camera.SetProjectionType(1)
                filename = str(img_root / v_folder / f"shape_{total_views+i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                display.View.Dump(str(img_root / v_folder / f"view_.png"))
                svr_imgs.append(img1)

            num_views = 256  # 任意数量的视角
            sphere_points = fibonacci_sphere_samples(num_views) * 3  # 乘以半径
            for i, point in enumerate(sphere_points):
                # 相机位置在球面上
                view = point

                # 计算看向原点的上方向（保持Z轴向上）
                # 需要更复杂的上方向计算来避免万向锁
                direction = -view / np.linalg.norm(view)  # 看向原点的方向

                # 构建相机坐标系
                right = np.cross([0, 0, 1], direction)
                right = right / np.linalg.norm(right)
                up = np.cross(direction, right)

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
                display.camera.SetFOVy(int(np.random.choice([30, 35, 40, 45, 50])))

                display.camera.SetProjectionType(0)
                filename = str(img_root / v_folder / f"shape_{total_views+total_views+i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                display.View.Dump(str(img_root / v_folder / f"view_.png"))
                svr_imgs.append(img1)

            sphere_points = fibonacci_sphere_samples(num_views) * 3  # 乘以半径
            for i, point in enumerate(sphere_points):
                # 相机位置在球面上
                view = point

                # 计算看向原点的上方向（保持Z轴向上）
                # 需要更复杂的上方向计算来避免万向锁
                direction = -view / np.linalg.norm(view)  # 看向原点的方向

                # 构建相机坐标系
                right = np.cross([0, 0, 1], direction)
                right = right / np.linalg.norm(right)
                up = np.cross(direction, right)

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
                display.camera.SetFOVy(int(np.random.choice([30, 35, 40, 45, 50])))

                display.camera.SetProjectionType(1)
                filename = str(img_root / v_folder / f"shape_{total_views + total_views + num_views + i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                display.View.Dump(str(img_root / v_folder / f"view_.png"))
                svr_imgs.append(img1)

            # Sketch
            sketch_imgs = []
            display.camera.SetEyeAndCenter(gp_Pnt(2., 2., 2.), gp_Pnt(0., 0., 0.))
            display.camera.SetUp(gp_Dir(0, 0, 1))
            display.camera.SetAspect(1)
            display.camera.SetFOVy(45)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))

            init_pos = np.array((2, 2, 2))
            up_pos = np.array((2, 2, 3))

            mat.SetAmbientColor(Quantity_Color(1, 1, 1, Quantity_TOC_RGB))
            display.camera.SetProjectionType(1)
            display.DisplayShape(shape, material=mat, update=True)
            display.camera.SetAspect(1)
            display.camera.SetFOVy(45)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))
            for i in range(total_views):
                angles = np.array([
                    i % x_steps,
                    i // x_steps % y_steps,
                    i // (x_steps * y_steps)
                ])
                matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
                view = (matrix @ init_pos.T).T
                up = (matrix @ up_pos.T).T

                # new_v = (matrix.T @ np.asarray(v).T).T
                # trimesh.Trimesh(vertices=new_v, faces=f).export(str(img_root / v_folder / f"mesh_{i}.ply"))

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))

                display.camera.SetProjectionType(0)
                filename = str(img_root / v_folder / f"wire_{i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                sketch_imgs.append(img1)

            for i in range(total_views):
                angles = np.array([
                    i % x_steps,
                    i // x_steps % y_steps,
                    i // (x_steps * y_steps)
                ])
                matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
                view = (matrix @ init_pos.T).T
                up = (matrix @ up_pos.T).T

                # new_v = (matrix.T @ np.asarray(v).T).T
                # trimesh.Trimesh(vertices=new_v, faces=f).export(str(img_root / v_folder / f"mesh_{i}.ply"))

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))

                display.camera.SetProjectionType(1)
                filename = str(img_root / v_folder / f"wire_{total_views+i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                sketch_imgs.append(img1)

            for i, point in enumerate(sphere_points):
                # 相机位置在球面上
                view = point

                # 计算看向原点的上方向（保持Z轴向上）
                # 需要更复杂的上方向计算来避免万向锁
                direction = -view / np.linalg.norm(view)  # 看向原点的方向

                # 构建相机坐标系
                right = np.cross([0, 0, 1], direction)
                right = right / np.linalg.norm(right)
                up = np.cross(direction, right)

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
                display.camera.SetFOVy(int(np.random.choice([30, 35, 40, 45, 50])))

                display.camera.SetProjectionType(0)
                filename = str(img_root / v_folder / f"wire_{total_views+total_views+i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                sketch_imgs.append(img1)

            for i, point in enumerate(sphere_points):
                # 相机位置在球面上
                view = point

                # 计算看向原点的上方向（保持Z轴向上）
                # 需要更复杂的上方向计算来避免万向锁
                direction = -view / np.linalg.norm(view)  # 看向原点的方向

                # 构建相机坐标系
                right = np.cross([0, 0, 1], direction)
                right = right / np.linalg.norm(right)
                up = np.cross(direction, right)

                display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
                display.camera.SetFOVy(int(np.random.choice([30, 35, 40, 45, 50])))

                display.camera.SetProjectionType(1)
                filename = str(img_root / v_folder / f"wire_{total_views+total_views+num_views+i}.png")
                data = display.GetImageData(224, 224)
                img = Image.frombytes("RGB", (224, 224), data)
                img1 = np.asarray(img)
                img1 = img1[::-1]
                #Image.fromarray(img1).save(filename)
                sketch_imgs.append(img1)

            # Multi view
            display.camera.SetEyeAndCenter(gp_Pnt(2., 2., 2.), gp_Pnt(0., 0., 0.))
            display.camera.SetUp(gp_Dir(0, 0, 1))
            display.camera.SetAspect(1)
            display.camera.SetFOVy(45)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))

            init_pos = np.array((2, 2, 2))
            up_pos = np.array((2, 2, 3))

            mat = Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_Silver)
            mat.SetReflectionModeOff(Graphic3d_TypeOfReflection.Graphic3d_TOR_SPECULAR)
            display.camera.SetProjectionType(1)
            display.DisplayShape(shape, material=mat, update=True)
            display.View.Dump(str(img_root / v_folder / f"view_.png"))
            mvr_imgs = []

            for j in range(8):
                if j == 0:
                    init_pos = np.array((2, 2, 2))
                elif j == 1:
                    init_pos = np.array((-2, 2, 2))
                elif j == 2:
                    init_pos = np.array((-2, -2, 2))
                elif j == 3:
                    init_pos = np.array((2, -2, 2))
                elif j == 4:
                    init_pos = np.array((2, 2, -2))
                elif j == 5:
                    init_pos = np.array((-2, 2, -2))
                elif j == 6:
                    init_pos = np.array((-2, -2, -2))
                elif j == 7:
                    init_pos = np.array((2, -2, -2))
                up_pos = init_pos + np.array((0, 0, 1))
                for i in range(64):
                    angles = np.array([
                        i % 4,
                        i // 4 % 4,
                        i // 16
                    ])
                    matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
                    view = (matrix @ init_pos.T).T
                    up = (matrix @ up_pos.T).T

                    # new_v = (matrix.T @ np.asarray(v).T).T
                    # trimesh.Trimesh(vertices=new_v, faces=f).export(str(img_root / v_folder / f"mesh_{i}.ply"))

                    display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
                    display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
                    filename = str(img_root / v_folder / f"mvr_{i}_{j}.png")
                    data = display.GetImageData(224, 224)
                    img = Image.frombytes("RGB", (224, 224), data)
                    img1 = np.asarray(img)
                    img1 = img1[::-1]
                    # Image.fromarray(img1).save(filename)
                    mvr_imgs.append(img1)

            svr_imgs = np.stack(svr_imgs, axis=0)
            mvr_imgs = np.stack(mvr_imgs, axis=0)
            sketch_imgs = np.stack(sketch_imgs, axis=0)
            data_dict["svr_imgs"] = svr_imgs
            data_dict["mvr_imgs"] = mvr_imgs
            data_dict["sketch_imgs"] = sketch_imgs

            np.savez_compressed(output_root / v_folder / "data.npz", **data_dict)
        except Exception as e:
            with open(output_root / "error.txt", "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(v_folder + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(v_folder + ": " + str(e))
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(e)
                shutil.rmtree(output_root / v_folder)

    for folder in single_loop_folder:
        with open(output_root / "single_loop.txt", "a") as f:
            f.write(folder + "\n")

    return


get_brep_ray = ray.remote(get_brep)

if __name__ == '__main__':
    total_ids = [item.strip() for item in open(data_split, "r").readlines()]

    exception_ids = []
    for file in exception_files:
        exception_ids += [item.strip() for item in open(file, "r").readlines()]
    exception_ids = list(set(exception_ids))

    num_original = len(total_ids)
    total_ids = list(set(total_ids) - set(exception_ids))
    total_ids.sort()
    total_ids = total_ids
    print("Total ids: {} -> {}".format(num_original, len(total_ids)))
    check_dir(npz_root)
    safe_check_dir(img_root)

    # single process
    if debug_id is not None:
        total_ids = [debug_id]
        get_brep(data_root, npz_root, total_ids)
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=15000,
                #num_cpus=1,
                #local_mode=True
        )
        batch_size = 1
        num_batches = len(total_ids) // batch_size + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(
                    get_brep_ray.remote(data_root,
                                        npz_root,
                                        total_ids[i * batch_size:min(len(total_ids), (i + 1) * batch_size)]))
        ray.get(tasks)
        print("Done")
