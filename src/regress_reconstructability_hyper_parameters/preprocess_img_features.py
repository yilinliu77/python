import math
import os
import pickle
import re
import time
from functools import partial, reduce
from multiprocessing import Pool

import cv2
import hydra
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import List, Tuple

from plyfile import PlyData, PlyElement
from torchvision.transforms import transforms
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats
from tqdm.contrib.concurrent import thread_map, process_map

from src.regress_reconstructability_hyper_parameters.preprocess_view_features import preprocess_data


def compute_features(v_root_path,
                     v_data
                     ):
    point_feature_path, img_paths, view = v_data
    point_path = os.path.join(v_root_path, "point_features", point_feature_path)

    point_features = []
    for id_view, view_path in enumerate(img_paths):
        pixel_position = torch.tensor(view[id_view, 6:8], dtype=torch.float32)
        pixel_position = torch.cat([pixel_position - 10 / 400, pixel_position + 10 / 400], dim=-1)
        pixel_position[pixel_position>1]=1
        pixel_position[pixel_position<0]=0
        # Get img features
        # Debug
        # img = cv2.resize(cv2.imread(view_path,cv2.IMREAD_UNCHANGED),(600,400))
        # img=np.asarray(img).copy()
        # test_pixel_position=pixel_position.numpy()
        # test_pixel_position[[0,2]]*=600
        # test_pixel_position[[1,3]]*=400
        # test_pixel_position=test_pixel_position.astype(np.int)
        # img[test_pixel_position[1]:test_pixel_position[3],test_pixel_position[0]:test_pixel_position[2],:3]=(255,0,0)
        # cv2.namedWindow("1",cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("1", img)
        # cv2.imshow("1", np.asarray(img))
        # cv2.waitKey(0)
        # Debug

        # Get the pixel features

        point_features.append((view_path, pixel_position, point_path))
    if len(point_features)==0:
        return
    return point_features

if __name__ == '__main__':
    import sys

    print("Using images, poses and sample points to compute img features")

    v_data_root = sys.argv[1]
    v_output_root = sys.argv[2]
    reconstructability_file_dir = os.path.join(v_data_root, "reconstructability")
    sample_points = os.path.join(v_data_root, "sampling_points.ply")
    img_dir = os.path.join(v_data_root, "images")
    img_rescale_size = (400, 600)

    print("First, compute the geometric features from the viewpoint to sample points")
    if not os.path.exists(v_output_root):
        os.mkdir(v_output_root)
    view, view_pair, point_attribute, point_features_path, img_paths = preprocess_data(
        reconstructability_file_dir,
        sample_points,
        img_dir
    )
    np.savez_compressed(os.path.join(v_output_root, "views"), view)
    np.savez_compressed(os.path.join(v_output_root, "view_paths"), point_features_path)
    np.savez_compressed(os.path.join(v_output_root, "img_paths"), img_paths)
    print("Pre-compute data done")

    # debug
    # view = np.load(os.path.join(output_root, "training_data/views.npz"))["arr_0"]
    # view_paths = np.load(os.path.join(output_root, "training_data/view_paths.npz"), allow_pickle=True)[
    #     "arr_0"]
    # debug

    # Compute view features
    print("Compute view features")
    if not os.path.exists(os.path.join(v_data_root, "view_features")):
        os.mkdir(os.path.join(v_data_root, "view_features"))
    transform = transforms.Compose([
        transforms.Resize(img_rescale_size),
        transforms.ToTensor(),
    ])
    from thirdparty.AA_RMVSNet.models.drmvsnet import AARMVSNet
    img_feature_extractor = AARMVSNet()
    state_dict = torch.load("thirdparty/AA_RMVSNet/model_release.ckpt")
    img_feature_extractor.load_state_dict(state_dict['model'], True)
    img_feature_extractor.requires_grad_(False)
    img_feature_extractor.eval()
    img_feature_extractor

    # All the image names
    total_view_paths = set()
    for item in img_paths:
        total_view_paths = total_view_paths.union(set(item))

    img_features_task = {}
    for item in total_view_paths:
        img_features_task[item]=[]

    num_points = img_paths.shape[0]
    output_feature = [None] * num_points
    for id_point, point_task in enumerate(tqdm(img_paths)):
        for id_view, item in enumerate(point_task):
            img_name = item
            pixel_pos = view[id_point][id_view][6:8]
            img_features_task[img_name].append((pixel_pos,(id_point, id_view)))
        output_feature[id_point] = np.zeros((len(point_task),32))

    img_features_dict = {}
    total_view_paths = list(total_view_paths)
    for id_view in tqdm(range(0, len(total_view_paths))):
        save_name = total_view_paths[id_view].split(".")[0]
        save_name = re.sub(r"\\", r"/", save_name)
        save_name = save_name.split("/")[-1] + ".npz"
        img_features_saved_path = os.path.join(v_data_root, "view_features", save_name)
        if os.path.exists(img_features_saved_path):
            img_features_dict[total_view_paths[id_view]]=np.load(img_features_saved_path)["arr_0"]
        else:
            img = Image.open(total_view_paths[id_view])
            img = transform(img)
            var = torch.var(img, dim=(1, 2), keepdim=True)
            mean = torch.mean(img, dim=(1, 2), keepdim=True)
            img_tensor = (img - mean) / (np.sqrt(var) + 0.00000001)
            img_features = img_feature_extractor.feature(img_tensor.unsqueeze(0)[:, :3])
            numpy_features = img_features.cpu().numpy()
            img_features_dict[total_view_paths[id_view]] = numpy_features
            np.savez(img_features_saved_path,numpy_features)

    print("Pre compute features")
    if not os.path.exists(os.path.join(v_output_root, "point_features")):
        os.mkdir(os.path.join(v_output_root, "point_features"))

    batch_size = 2560
    for img_name in tqdm(img_features_task):
        img_feature = img_features_dict[img_name]
        positions = list(map(lambda x:x[0],img_features_task[img_name]))
        output_positions = list(map(lambda x:x[1],img_features_task[img_name]))
        for i in range(0,len(positions),batch_size):
            positions_item = np.array(positions[i:min(len(positions),i+batch_size)])
            output_positions_item = np.array(output_positions[i:min(len(positions),i+batch_size)])
            positions_item=np.concatenate([positions_item - 10 / 400, positions_item + 10 / 400],axis=1)
            positions_item = np.clip(positions_item,0,1)
            positions_item[:,[0,2]] = positions_item[:,[0,2]] * 600
            positions_item[:,[1,3]] = positions_item[:,[1,3]] * 400
            pixel_position_features = torchvision.ops.roi_align(
                torch.tensor(img_feature),
                [torch.tensor(positions_item)],
                1).squeeze(
                -1).squeeze(-1).cpu().numpy()
            for id_batch, out_pos in enumerate(output_positions_item):
                output_feature[out_pos[0]][out_pos[1]] = pixel_position_features[id_batch]

    thread_map(lambda x: np.savez(os.path.join(v_output_root, "point_features",x[0]),x[1]),zip(point_features_path,output_feature))
    print("Pre compute done")
