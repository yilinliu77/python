import os

from scipy.stats import stats

from src.regress_reconstructability_hyper_parameters.dataset import Regress_hyper_parameters_dataset, \
    Regress_hyper_parameters_dataset_with_imgs, Regress_hyper_parameters_dataset_with_imgs_with_truncated_error
from src.regress_reconstructability_hyper_parameters.model import Regress_hyper_parameters_Model, Brute_force_nn, \
    Correlation_nn

import numpy as np


def print_distribution(v_array: np.ndarray, v_title="") -> None:
    print("{:30}: Mean: {:.4f}; Min: {:.4f}; Max: {:.4f}; 25%: {:.4f}; 50%: {:.4f}; 75%: {:.4f};".format(
        v_title,
        np.mean(v_array),
        np.quantile(v_array, 0),
        np.quantile(v_array, 1),
        np.quantile(v_array, 0.25),
        np.quantile(v_array, 0.5),
        np.quantile(v_array, 0.75),
    ))


if __name__ == '__main__':
    v_root = r"D:\Projects\Reconstructability\training_data\v6"

    summary_spearman = {
        'fine': {
            "acc": 0,
            "com": 0,
            "num": 0
        },
        'preview': {
            "acc": 0,
            "com": 0,
            "num": 0
        },
        'inter': {
            "acc": 0,
            "com": 0,
            "num": 0
        },
        'coarse': {
            "acc": 0,
            "com": 0,
            "num": 0
        },
    }

    summary_variable = {
        "view_theta":[],
        "view_phi":[],
        "view_distance":[],
        "view_angle_2_normal":[],
        "view_angle_2_direction":[],
        "gt_accuracy":[],
        "gt_completeness":[],
    }

    for item in os.listdir(v_root):
        # if "chengbao" not in item:
        #     continue
        # if "fine" not in item and "preview" not in item:
        #     continue

        if item[-1] != "0":
            continue
        params = {
            "model": {
                "involve_img": False,
                "view_mean_std":[0,0,0,0,0,1,1,1,1,1],
                "error_mean_std":[0,0,1,1],
            }
        }

        dataset_item = Regress_hyper_parameters_dataset_with_imgs(os.path.join(v_root, item), params, "test")
        print("========================= {:20} =========================".format(item))

        if "fine" in item:
            key = "fine"
        elif "preview" in item:
            key = "preview"
        elif "inter" in item:
            key = "inter"
        elif "coarse" in item:
            key = "coarse"
        else:
            raise ""

        print("============= View features =============")
        views = dataset_item.views.astype(np.float32)
        view_num = views[:, :, 0].sum(axis=-1)
        valid_flag = views[:, :, 0] == 1
        view_theta = views[valid_flag][:, 1].reshape(-1)
        view_phi = views[valid_flag][:, 2].reshape(-1)
        view_distance = views[valid_flag][:, 3].reshape(-1)
        view_angle_2_normal = views[valid_flag][:, 4].reshape(-1)
        view_angle_2_direction = views[valid_flag][:, 5].reshape(-1)
        print_distribution(view_num, "view_num")
        print_distribution(view_theta, "view_theta")
        print_distribution(view_phi, "view_phi")
        print_distribution(view_distance, "view_distance")
        print_distribution(view_angle_2_normal, "view_angle_2_normal")
        print_distribution(view_angle_2_direction, "view_angle_2_direction")

        summary_variable["view_theta"].append(view_theta)
        summary_variable["view_phi"].append(view_phi)
        summary_variable["view_distance"].append(view_distance)
        summary_variable["view_angle_2_normal"].append(view_angle_2_normal)
        summary_variable["view_angle_2_direction"].append(view_angle_2_direction)

        print("\n============= Error =============")
        point_attribute = dataset_item.point_attribute.astype(np.float32)
        smith_recon = point_attribute[:, 0]
        gt_accuracy = point_attribute[:, 1]
        gt_completeness = point_attribute[:, 2]
        acc_mask = gt_accuracy != -1
        com_mask = gt_completeness != -1
        assert np.all((1 - point_attribute[:, 6]).astype(np.bool_) == (gt_accuracy != -1))
        acc_spearmanr = stats.spearmanr(smith_recon[acc_mask], gt_accuracy[acc_mask])[0]
        com_spearmanr = stats.spearmanr(smith_recon[com_mask], gt_completeness[com_mask])[0]

        summary_variable["gt_accuracy"].append(gt_accuracy[acc_mask])
        summary_variable["gt_completeness"].append(gt_completeness[com_mask])

        print_distribution(smith_recon, "smith_recon")
        print_distribution(gt_accuracy[acc_mask], "gt_accuracy")
        print_distribution(gt_completeness[gt_completeness != -1], "gt_completeness")
        print("Spearman factor of accuracy: {:.2f}; {}/{} points".format(acc_spearmanr, acc_mask.sum(),
                                                                         point_attribute.shape[0]))
        print("Spearman factor of completeness: {:.2f}; {}/{} points".format(com_spearmanr, com_mask.sum(),
                                                                             point_attribute.shape[0]))

        summary_spearman[key]["acc"]+=acc_spearmanr
        summary_spearman[key]["com"]+=com_spearmanr
        summary_spearman[key]["num"]+=1
        print("==============================================================\n")

        pass

    for proxy in summary_spearman:
        summary_spearman[proxy]["acc"] /=summary_spearman[proxy]["num"] + 1e-6
        summary_spearman[proxy]["com"] /=summary_spearman[proxy]["num"] + 1e-6
    print(summary_spearman)

    for item in summary_variable:
        summary_variable[item] = np.concatenate(summary_variable[item])
        print_distribution(summary_variable[item], item)
        print("{:20}; Mean: {:.4f}; Std: {:.4f}".format(
            item,
            np.mean(summary_variable[item]),
            np.std(summary_variable[item]))
        )
