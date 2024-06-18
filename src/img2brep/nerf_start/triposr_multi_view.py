import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import open3d.cpu.pybind.io
import rembg
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append("thirdparty/triposr")

from thirdparty.triposr.tsr.system import TSR
from thirdparty.triposr.tsr.utils import remove_background, resize_foreground, save_video, scale_tensor


###
# Input: a folder that contains a transform.json file and a list of images
# Output: Specified by the output_dir variable (output/triposr)
###

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()

chunk_size = 8192
is_render = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)

args = parser.parse_args()

output_dir = "output/triposr"
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"

timer.start("Initializing model")
model = TSR.from_pretrained(
    "thirdparty/triposr/stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []

rembg_session = rembg.new_session()

for dir in args.dir:
    files = os.listdir(dir)
    print(files)

    dataset = json.loads(open(os.path.join(dir, "transform.json")).read())
    frames = dataset["frames"]

    for _, frame in enumerate(frames):
        image = Image.open(os.path.join(dir, frame["file_path"]))
        image = remove_background(image, rembg_session)
        # image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        # image[image[:, :, 3] == 0] = 1
        # image = image[:,:,:3]
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        image.save(os.path.join(output_dir, frame["file_path"]))
        images.append(image)
    timer.end("Processing images")

    for i, frame in enumerate(frames):
        image = images[i]
        extrinsic = np.asarray(frame["transform_matrix"]).astype(np.float32)
        logging.info(f"Running image {i + 1}/{len(images)} ...")

        timer.start("Running model")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        timer.end("Running model")

        if is_render:
            timer.start("Rendering")
            render_images = model.render(scene_codes, n_views=1, return_type="pil")
            render_images[0][0].save(os.path.join(output_dir, r"render_000.png"))
            # for ri, render_image in enumerate(render_images[0]):
            #     render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))
            # save_video(
            #     render_images[0], os.path.join(output_dir, f"render.mp4"), fps=30
            # )
            timer.end("Rendering")

        transform_matrix = np.asarray(frame["transform_matrix"]).astype(np.float32)
        scale_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        w2c = scale_matrix @ np.linalg.inv(transform_matrix)

        radius = 3.
        pos = transform_matrix[:3, 3]
        relev = np.arcsin(pos[2] / radius)
        razim = np.arctan2(pos[1], pos[0])
        rotation = Rotation.from_euler("zyx", [-razim, relev, 0], degrees=False).as_matrix()

        timer.start("Exporting mesh")
        meshes = model.extract_mesh(scene_codes, resolution=256, threshold=25.0)
        original_vertices = (meshes[0].vertices).copy()  # [-0.87, 0.87]
        vertices = original_vertices * 2
        vertices = (np.linalg.inv(rotation) @ vertices.T).T

        meshes[0].vertices = vertices
        meshes[0].export(os.path.join(output_dir, frame["file_path"].split(".")[0] + ".obj"))
        timer.end("Exporting mesh")

        def get_feature(model, scene_codes):
            with torch.no_grad():
                net_out = model.renderer.query_triplane(
                    model.decoder,
                    scale_tensor(
                        model.isosurface_helper.grid_vertices.to(scene_codes.device),
                        model.isosurface_helper.points_range,
                        (-model.renderer.cfg.radius, model.renderer.cfg.radius),
                    ),
                    scene_codes[0],
                )
                features = torch.cat([net_out["features"], net_out["density"]], dim=1)
                features_cpu = features.reshape(256, 256, 256, 4).cpu().numpy()
                np.save(os.path.join(output_dir, frame["file_path"].split(".")[0] + ".npy"), features_cpu)

        timer.start("Exporting feature")
        get_feature(model, scene_codes)
        timer.end("Exporting feature")
