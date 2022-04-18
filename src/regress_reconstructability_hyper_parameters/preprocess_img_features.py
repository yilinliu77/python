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
from ternausnet.models import UNet16, UNet11
from torch import nn
from torchvision.models import resnet, resnet50, resnet101, resnet34
from torchvision.transforms import transforms
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats
from tqdm.contrib.concurrent import thread_map, process_map

from shared.common_utils import debug_imgs
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

def compute_view_features(v_model, v_view_path: str):
    save_name = v_view_path.split(".")[0]
    save_name = re.sub(r"\\", r"/", save_name)
    save_name = save_name.split("/")[-1] + ".npy"
    img_features_saved_path = os.path.join(v_data_root, "view_features", save_name)
    if os.path.exists(img_features_saved_path):
        img_features_dict[v_view_path] = img_features_saved_path
    else:
        img = np.asarray(Image.open(v_view_path)).copy()[:, :, :3]
        img_tensor = pre_transform(img)
        with torch.no_grad():
            img_features = v_model(
                img_tensor.unsqueeze(0).to(device))
        numpy_features = img_features.cpu().numpy()
        # original_img = np.asarray(Image.open(total_view_paths[id_view])).copy()[:,:,:3]
        # resized_img = img.astype(np.uint8)
        # transformed_img = img_tensor
        # featured_img = np.zeros([240,360,1],dtype=np.float32)
        #
        # for y in range(6):
        #     for x in range(6):
        #         c = y*6+x
        #         if c>=32:
        #             break
        #         start_y = y * 40
        #         start_x = x * 60
        #         featured_img[start_y:start_y+40,start_x:start_x+60,0] = cv2.resize(numpy_features[0,c][:,:,np.newaxis],(60,40))
        #
        # debug_imgs([original_img])
        # debug_imgs([resized_img])
        # debug_imgs([transformed_img])
        # debug_imgs([featured_img])
        img_features_dict[v_view_path] = img_features_saved_path
        np.save(img_features_saved_path, numpy_features)

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
    np.save(os.path.join(v_output_root, "views"), view)
    np.save(os.path.join(v_output_root, "view_paths"), point_features_path)
    np.save(os.path.join(v_output_root, "img_paths"), img_paths)
    print("Pre-compute data done")

    # debug
    # view = np.load(os.path.join(v_output_root, "views.npy"))
    # point_features_path = np.load(os.path.join(v_output_root, "view_paths.npy"), allow_pickle=True)
    # img_paths = np.load(os.path.join(v_output_root, "img_paths.npy"), allow_pickle=True)
    # debug

    # Compute view features
    print("Compute view features")
    if not os.path.exists(os.path.join(v_data_root, "view_features")):
        os.mkdir(os.path.join(v_data_root, "view_features"))

    device = "cpu"
    device = "cuda"

    if True:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_rescale_size),
        ])
        def pre_transform(v_img):
            img = np.asarray(transform(v_img), dtype=np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img_tensor = (img - mean) / (np.sqrt(var))
            return torch.tensor(img_tensor.transpose([2,0,1]))

        from thirdparty.AA_RMVSNet.models.drmvsnet import AARMVSNet
        img_feature_extractor = AARMVSNet()
        state_dict = torch.load("thirdparty/AA_RMVSNet/model_release.ckpt")
        img_feature_extractor.load_state_dict(state_dict['model'], True)
        img_feature_extractor.requires_grad_(False)
        img_feature_extractor = img_feature_extractor.feature
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480,640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        def pre_transform(v_img):
            img = transform(v_img)
            return img
        img_feature_extractor = UNet16(pretrained=True, num_classes=256)

    img_feature_extractor.eval()
    img_feature_extractor = img_feature_extractor.to(device)

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
            # img = np.asarray(Image.open(img_name))
            # x1 = max(0,pixel_pos[0]-0.025)*6000
            # x2 = min(1,pixel_pos[0]+0.025)*6000
            # y1 = max(0,pixel_pos[1]-0.025)*4000
            # y2 = min(1,pixel_pos[1]+0.025)*4000
            # cv2.rectangle(img,
            #               (int(x1),int(y1)),
            #               (int(x2),int(y2)),
            #               (255,0,0),int(10))
            # debug_imgs([img[:,:,:3]])
        output_feature[id_point] = np.zeros((len(point_task),256))

    img_features_dict = {}
    total_view_paths = list(total_view_paths)

    with torch.no_grad():
        thread_map(partial(compute_view_features,img_feature_extractor),total_view_paths,max_workers=8)

    print("Pre compute features")
    if not os.path.exists(os.path.join(v_output_root, "point_features")):
        os.mkdir(os.path.join(v_output_root, "point_features"))

    batch_size = 2560
    for img_name in tqdm(img_features_task):
        img_feature = np.load(img_features_dict[img_name])
        positions = list(map(lambda x:x[0],img_features_task[img_name]))
        output_positions = list(map(lambda x:x[1],img_features_task[img_name]))
        for i in range(0,len(positions),batch_size):
            positions_item = np.array(positions[i:min(len(positions),i+batch_size)])
            output_positions_item = np.array(output_positions[i:min(len(positions),i+batch_size)])
            positions_item=np.concatenate([positions_item - 10 / 400, positions_item + 10 / 400],axis=1)
            positions_item = np.clip(positions_item,0,1)
            positions_item[:,[0,2]] = positions_item[:,[0,2]] * img_feature.shape[3]
            positions_item[:,[1,3]] = positions_item[:,[1,3]] * img_feature.shape[2]
            pixel_position_features = torchvision.ops.roi_align(
                torch.tensor(img_feature),
                [torch.tensor(positions_item)],
                1).squeeze(
                -1).squeeze(-1).cpu().numpy()
            for id_batch, out_pos in enumerate(output_positions_item):
                output_feature[out_pos[0]][out_pos[1]] = pixel_position_features[id_batch]

    thread_map(lambda x: np.savez(os.path.join(v_output_root, "point_features",x[0]),x[1]),zip(point_features_path,output_feature))
    print("Pre compute done")
