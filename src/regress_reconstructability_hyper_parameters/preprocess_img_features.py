import os
import pickle
from functools import partial
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


def compute_view_features(max_num_view,valid_views_flag,reconstructabilities,v_img_dir,
                          views,views_pair,view_paths,
                          v_file):
    real_index = int(v_file.split("\\")[-1][:-4])

    raw_data = [item.strip() for item in open(v_file).readlines()]
    num_views = int(raw_data[0])
    if num_views > max_num_view - 1:
        raise
    # Read views
    for i_view in range(num_views):
        view_data = raw_data[2 + i_view].split(",")
        img_name = view_data[0]
        view_to_point = np.array(view_data[1:4], dtype=np.float16)
        distance_ratio = float(view_data[4])
        angle_to_normal_ratio = float(view_data[5])
        angle_to_direction_ratio = float(view_data[6])
        pixel_pos_x = float(view_data[7])
        pixel_pos_y = float(view_data[8])
        if v_img_dir:
            img_path = os.path.join(v_img_dir, img_name + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(v_img_dir, img_name + ".jpg")
            if not os.path.exists(img_path):
                raise
            view_paths[real_index].append(img_path)
            # cv2.imshow("",cv2.resize(cv2.imread(img_path),(600,400)))
            # cv2.waitKey()
        views[real_index][i_view][0] = 1
        views[real_index][i_view][1] = view_to_point[0]
        views[real_index][i_view][2] = view_to_point[1]
        views[real_index][i_view][3] = view_to_point[2]
        views[real_index][i_view][4] = distance_ratio
        views[real_index][i_view][5] = angle_to_normal_ratio
        views[real_index][i_view][6] = angle_to_direction_ratio
        views[real_index][i_view][7] = pixel_pos_x
        views[real_index][i_view][8] = pixel_pos_y

        """
        debug
        """
        # cv2.namedWindow("1", cv2.WINDOW_AUTOSIZE)
        # img = cv2.resize(cv2.imread(view_paths[real_index][i_view]),(600,400))
        # pt = np.array([pixel_pos_x*600,pixel_pos_y*400],dtype=np.int32)
        # img[pt[1]-5:pt[1]+5,pt[0]-5:pt[0]+5]=(0,0,255)
        # cv2.imshow("1", img)
        # cv2.waitKey()
        """
        debug done
        """
        continue


def preprocess_data(v_root: str, v_error_point_cloud: str, v_img_dir: str = None) -> (np.ndarray, np.ndarray):
    print("Read point cloud")
    with open(v_error_point_cloud, "rb") as f:
        plydata = PlyData.read(f)
        num_points = plydata['vertex']['x'].shape[0]

    files = [os.path.join(v_root, item) for item in os.listdir(v_root)]
    files = list(filter(lambda item: ".txt" in item, files))
    files = sorted(files, key=lambda item: int(item.split("\\")[-1][:-4]))

    # Find max view numbers
    def compute_max_view_number(v_file):
        raw_data = [item.strip() for item in open(v_file).readlines()]
        num_views = int(raw_data[0])
        return num_views
    num_view_list = thread_map(compute_max_view_number, files, max_workers=8)
    # num_view_list = process_map(compute_max_view_number, files, max_workers=8,chunksize=1)
    max_num_view = max(num_view_list) + 1
    print("Read attribute with max view: ", max_num_view)
    view_paths = [[] for _ in range(num_points)]
    views = np.zeros((num_points, max_num_view, 9), dtype=np.float16)
    views_pair = np.zeros((num_points, max_num_view, max_num_view, 3), dtype=np.float16)
    valid_views_flag = [False for _ in range(num_points)]
    reconstructabilities = [0 for _ in range(num_points)]

    thread_map(partial(compute_view_features,
                       max_num_view,valid_views_flag,reconstructabilities,v_img_dir,views,views_pair,view_paths),
               files,max_workers=10)

    return views, view_paths

def compute_features(v_root_path,v_view_paths,v_view_attribute,img_features_dict,
                     id_point):
    point_path = os.path.join(v_root_path, "point_features", str(id_point) + ".npz")

    point_features = []
    for id_view, view_path in enumerate(v_view_paths[id_point]):
        pixel_position = torch.tensor(v_view_attribute[id_point, id_view, 7:9], dtype=torch.float32)
        pixel_position = torch.cat([pixel_position - 20 / 400, pixel_position + 20 / 400], dim=-1)
        pixel_position[pixel_position>1]=1
        pixel_position[pixel_position<0]=0
        # Get img features
        img_features=img_features_dict[view_path]
        # Debug
        # img = Image.open(view_path).resize((600,400))
        # img=np.asarray(img).copy()
        # test_pixel_position=pixel_position.numpy()
        # test_pixel_position[[0,2]]*=600
        # test_pixel_position[[1,3]]*=400
        # test_pixel_position=test_pixel_position.astype(np.int)
        # img[test_pixel_position[1]:test_pixel_position[3],test_pixel_position[0]:test_pixel_position[2],:3]=(255,0,0)
        # cv2.namedWindow("1",cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("1", img)
        # # cv2.imshow("1", np.asarray(img))
        # cv2.waitKey(0)
        # Debug

        # Get the pixel features
        pixel_position_features = torchvision.ops.ps_roi_align(
            torch.tensor(img_features),
            [pixel_position.unsqueeze(0)],
            1).squeeze(
            -1).squeeze(-1)
        point_features.append(pixel_position_features)
    if len(point_features)==0:
        return
    point_features = torch.cat(point_features, dim=0)
    np.savez(point_path, point_features.cpu().numpy())

def pre_compute_img_features(v_view_paths: List[str], img_features_dict, v_root_path, v_view_attribute):
    if not os.path.exists(os.path.join(v_root_path, "point_features")):
        os.mkdir(os.path.join(v_root_path, "point_features"))

    with torch.no_grad():
        r = thread_map(partial(compute_features, v_root_path, v_view_paths, v_view_attribute, img_features_dict),
                       range(len(v_view_paths)), max_workers=1)
        # for v_id in tqdm(range(len(v_view_paths))):
        #     partial(compute_features, v_root_path, v_view_paths, v_view_attribute, img_features_dict)(v_id)
    return


if __name__ == '__main__':
    import sys

    print("Using images, poses and sample points to compute img features")

    output_root = sys.argv[1]
    reconstructability_file_dir = os.path.join(output_root, "reconstructability")
    sample_points = os.path.join(output_root, "sample_points_proxy.ply")
    img_dir = os.path.join(output_root, "images")
    img_rescale_size = (400, 600)

    print("First, compute the geometric features from the viewpoint to sample points")
    if not os.path.exists(os.path.join(output_root, "training_data")):
        os.mkdir(os.path.join(output_root, "training_data"))
    view, view_paths = preprocess_data(
        reconstructability_file_dir,
        sample_points,
        img_dir
    )
    np.savez_compressed(os.path.join(output_root, "training_data/views"), view)
    np.savez_compressed(os.path.join(output_root, "training_data/view_paths"), view_paths)
    print("Pre-compute data done")

    # debug
    # view = np.load(os.path.join(output_root, "training_data/views.npz"))["arr_0"]
    # view_paths = np.load(os.path.join(output_root, "training_data/view_paths.npz"), allow_pickle=True)[
    #     "arr_0"]
    # debug

    # Compute view features
    print("Compute view features")
    if not os.path.exists(os.path.join(output_root, "training_data/view_features")):
        os.mkdir(os.path.join(output_root, "training_data/view_features"))
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
    img_features_dict = {}

    total_view_paths = set()
    for item in view_paths:
        total_view_paths = total_view_paths.union(set(item))
    total_view_paths = list(total_view_paths)
    for id_view in tqdm(range(0, len(total_view_paths))):
        save_name = total_view_paths[id_view].split(".")[0].split("\\")[-1] + ".npz"
        img_features_saved_path = os.path.join(os.path.join(output_root, "training_data"), "view_features", save_name)
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
    pre_compute_img_features(view_paths, img_features_dict, os.path.join(output_root, "training_data"), view)
    print("Pre compute done")
