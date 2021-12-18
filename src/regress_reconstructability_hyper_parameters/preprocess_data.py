import os
import pickle

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


def preprocess_data(v_root: str, v_error_point_cloud: str, v_img_dir: str = None) -> (np.ndarray, np.ndarray):
    print("Read point cloud")
    with open(v_error_point_cloud, "rb") as f:
        plydata = PlyData.read(f)
        if v_img_dir:
            x = plydata['vertex']['x'].copy()
            y = plydata['vertex']['y'].copy()
            z = plydata['vertex']['z'].copy()
            max_error_list = plydata['vertex']['max_error'].copy()
            sum_error_list = plydata['vertex']['sum_error'].copy()
            num_list = plydata['vertex']['num'].copy()
            avg_error = sum_error_list / (num_list + 1e-6)
            x_dim = (np.max(x) - np.min(x)) / 2
            y_dim = (np.max(y) - np.min(y)) / 2
            z_dim = (np.max(z) - np.min(z)) / 2
            max_dim = max(x_dim, y_dim, z_dim)
            np.savez(os.path.join(v_root,"../data_centralize"),np.array([np.mean(x),np.mean(y),np.mean(z),max_dim]))
            x = (x - np.mean(x)) / max_dim
            y = (y - np.mean(y)) / max_dim
            z = (z - np.mean(z)) / max_dim
            error_list = np.stack([max_error_list, avg_error, x, y, z, max_error_list==0], axis=1)
        else:
            error_list = plydata['vertex']['error'].copy()

    files = [os.path.join(v_root, item) for item in os.listdir(v_root)]
    files = list(filter(lambda item: ".txt" in item, files))
    files = sorted(files, key=lambda item: int(item.split("\\")[-1][:-4]))

    num_points = error_list.shape[0]
    # Find max view numbers
    max_num_view = 0
    for id_file, file in enumerate(tqdm(files)):
        raw_data = [item.strip() for item in open(file).readlines()]
        num_views = int(raw_data[0])
        max_num_view = max(max_num_view, num_views)
    max_num_view += 1
    print("Read attribute with max view: ", max_num_view)
    view_paths = [[] for _ in range(num_points)]
    views = np.zeros((num_points, max_num_view, 9), dtype=np.float16)
    views_pair = np.zeros((num_points, max_num_view, max_num_view, 3), dtype=np.float16)
    valid_views_flag = [False for _ in range(num_points)]
    reconstructabilities = [0 for _ in range(num_points)]
    for file in tqdm(files):
        real_index = int(file.split("\\")[-1][:-4])

        raw_data = [item.strip() for item in open(file).readlines()]
        num_views = int(raw_data[0])
        if num_views > max_num_view - 1:
            raise
        reconstructability = float(raw_data[1])
        if reconstructability != 0:
            valid_views_flag[real_index] = True
        reconstructabilities[real_index] = reconstructability
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

        # Read view pair
        cur_iter = 0
        for i_view1 in range(max_num_view):
            for i_view2 in range(i_view1 + 1, max_num_view):
                if i_view1 >= num_views or i_view2 >= num_views:
                    continue
                view_pair_data = raw_data[2 + num_views + cur_iter].split(",")
                alpha_ratio = float(view_pair_data[0])
                relative_distance_ratio = float(view_pair_data[1])
                views_pair[real_index][i_view1][i_view2][0] = 1
                views_pair[real_index][i_view1][i_view2][1] = alpha_ratio
                views_pair[real_index][i_view1][i_view2][2] = relative_distance_ratio
                cur_iter += 1

    # valid_flag = np.logical_and(np.array(valid_views_flag), error_list[:, 2] != 0)
    valid_flag = np.array(valid_views_flag)

    views = views[valid_flag]
    views_pair = views_pair[valid_flag]
    view_paths = np.array(view_paths)[valid_flag]
    reconstructabilities = np.asarray(reconstructabilities)[valid_flag]
    usable_indices = np.triu_indices(max_num_view, 1)
    views_pair = views_pair[:, usable_indices[0], usable_indices[1]]

    error_list = error_list[valid_flag]
    point_attribute = np.concatenate([reconstructabilities[:, np.newaxis], error_list], axis=1).astype(np.float16)
    print("Totally {} points; {} points has error 0; {} points has 0 reconstructability, {} in it is not visible; Output {} points".format(
        num_points,
        error_list[:, -1].sum(),
        num_points-valid_flag.sum(),
        num_points-len(files),
        point_attribute.shape[0]
    ))
    return views, views_pair, point_attribute, view_paths


def pre_compute_img_features(v_view_paths: List[str], v_img_size, v_root_path, v_view_attribute):
    transform = transforms.Compose([
        transforms.Resize(v_img_size),
        transforms.ToTensor(),
    ])

    from thirdparty.AA_RMVSNet.models.drmvsnet import AARMVSNet
    img_feature_extractor = AARMVSNet()
    state_dict = torch.load("thirdparty/AA_RMVSNet/model_release.ckpt")
    img_feature_extractor.load_state_dict(state_dict['model'], True)
    img_feature_extractor.requires_grad_(False)
    img_feature_extractor.eval()
    img_feature_extractor.cuda()
    img_features_dict = {}
    if not os.path.exists(os.path.join(v_root_path, "point_features")):
        os.mkdir(os.path.join(v_root_path, "point_features"))
    if not os.path.exists(os.path.join(v_root_path, "view_features")):
        os.mkdir(os.path.join(v_root_path, "view_features"))

    with torch.no_grad():
        for id_point, point in enumerate(tqdm(v_view_paths)):
            point_path = os.path.join(v_root_path, "point_features", str(id_point) + ".npz")
            if os.path.exists(point_path):
                continue
            point_features = []
            for id_view, view_path in enumerate(point):
                pixel_position = torch.tensor(v_view_attribute[id_point, id_view, 7:9], dtype=torch.float32)
                pixel_position = torch.cat([pixel_position - 5 / 400, pixel_position + 5 / 400], dim=-1)

                # Get img features
                img_features = None
                item_name = view_path.split(".")[0].split("\\")[-1] + ".npz"
                img_features_saved_path = os.path.join(v_root_path, "view_features", item_name)
                if os.path.exists(img_features_saved_path):
                    # Enable cache, but potent to pose memory leak. Can be use in the small scene
                    # if img_features_saved_path not in img_features_dict:
                    #     img_features_dict[img_features_saved_path] = torch.tensor(
                    #         np.load(img_features_saved_path)["arr_0"], dtype=torch.float32).cuda()
                    # img_features = img_features_dict[img_features_saved_path]
                    # Disable cache, slow
                    img_features_dict[img_features_saved_path] = torch.tensor(
                        np.load(img_features_saved_path)["arr_0"], dtype=torch.float32).cuda()
                else:
                    img = Image.open(view_path)
                    img = transform(img)
                    var = torch.var(img, dim=(0, 1), keepdim=True)
                    mean = torch.mean(img, dim=(0, 1), keepdim=True)
                    img = (img - mean) / (np.sqrt(var) + 0.00000001)
                    img_features = img_feature_extractor.feature(img[:3].unsqueeze(0).cuda())
                    np.savez_compressed(img_features_saved_path, img_features.cpu().numpy())

                # Get the pixel features
                pixel_position_features = torchvision.ops.ps_roi_align(
                    img_features,
                    [pixel_position.cuda().unsqueeze(0)],
                    1).squeeze(
                    -1).squeeze(-1)
                point_features.append(pixel_position_features)

            point_features = torch.cat(point_features, dim=0)
            np.savez(point_path, point_features.cpu().numpy())


if __name__ == '__main__':
    import sys

    output_root = sys.argv[1]
    reconstructability_file_dir = os.path.join(output_root, "reconstructability")
    error_point_cloud_dir = os.path.join(output_root, "accuracy_projected.ply")
    img_dir = os.path.join(output_root, "images")
    img_rescale_size = (600, 400)

    if not os.path.exists(os.path.join(output_root, "training_data")):
        os.mkdir(os.path.join(output_root, "training_data"))
    view, view_pair, point_attribute, view_paths = preprocess_data(
        reconstructability_file_dir,
        error_point_cloud_dir,
        img_dir
    )
    np.savez_compressed(os.path.join(output_root, "training_data/views"), view)
    np.savez_compressed(os.path.join(output_root, "training_data/view_pairs"), view_pair)
    np.savez_compressed(os.path.join(output_root, "training_data/point_attribute"), point_attribute)
    np.savez_compressed(os.path.join(output_root, "training_data/view_paths"), view_paths)
    print("Pre-compute data done")

    print("Pre compute features")
    pre_compute_img_features(view_paths, img_rescale_size, os.path.join(output_root, "training_data"), view)
    print("Pre compute done")
