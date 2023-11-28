import importlib
import os.path
from pathlib import Path

import hydra
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.common_utils import *
from src.neural_bsp.abc_hdf5_dataset import ABC_test_mesh, generate_coords

from src.neural_bsp.model import de_normalize_angles, de_normalize_udf

#
# Test both mesh udf and udc udf in the testset
#

@hydra.main(config_name="test_model.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    # Predefined variables
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))
    output_dir = Path(v_cfg["dataset"]["output_dir"])

    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    ps = 32
    query_points = generate_coords(res)

    # Dataset
    batch_size = v_cfg["trainer"]["batch_size"]
    mod = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
    dataset = getattr(mod, v_cfg["dataset"]["dataset_name"])(
        v_cfg["dataset"],
        res,
        batch_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    # check_dir(Path(v_cfg["dataset"]["root"]) / "prediction" / v_cfg["dataset"]["type"])
    check_dir(output_dir)
    data_root = dataset.data_root
    type = dataset.type

    # Load model
    mod = importlib.import_module('src.neural_bsp.model')
    model = getattr(mod, v_cfg["model"]["model_name"])(
        v_cfg["model"]
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    # tasks = tasks[9:10]
    # Start inference
    bar = tqdm(total=len(dataloader))
    precisions = []
    recalls = []
    f1s = []
    for data in dataloader:
        prefix = data[0][0]
        batched_data = [item[0] for item in data[1]]
        gt_flags = data[2][0].cuda()
        mesh_udf = data[3][0].numpy()

        bar.set_description(prefix)

        predictions = []
        for item in batched_data:
            feat = item.cuda().permute((0, 4, 1, 2, 3)).unsqueeze(1)
            feat = [(feat, torch.zeros(feat.shape[0])), None]
            prediction = model(feat, False).reshape(-1, ps, ps, ps)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0).reshape(15,15,15,32,32,32)
        predictions = predictions[:, :, :, 8:24, 8:24, 8:24].permute(0, 3, 1, 4, 2, 5).reshape(240, 240, 240)
        # if True:
        #     flags = -np.ones((res, res, res), dtype=np.float32) * 999
        #     x_start = y_start = z_start = 0
        #     while True:
        #         flags[
        #         x_start:x_start + ps,
        #         y_start:y_start + ps,
        #         z_start:z_start + ps
        #         ] = np.maximum(prediction.cpu().numpy(), flags[
        #                                                  x_start:x_start + ps,
        #                                                  y_start:y_start + ps,
        #                                                  z_start:z_start + ps
        #                                                  ])
        #
        #         z_start += ps_2
        #         if z_start + ps > res:
        #             z_start = 0
        #             y_start += ps_2
        #         if y_start + ps > res:
        #             y_start = 0
        #             x_start += ps_2
        #         if x_start + ps > res:
        #             break
        #
        #     final_flags = flags
        # else:
        #     flags = []
        #     x_start = y_start = z_start = 0
        #     while True:
        #         feat = mesh_udf[x_start:x_start + ps, y_start:y_start + ps, z_start:z_start + ps, :num_features]
        #         feat = torch.from_numpy(feat).cuda().permute(3, 0, 1, 2).unsqueeze(0).unsqueeze(0)
        #         feat = [(feat, torch.zeros(feat.shape[0])), None]
        #
        #         prediction = model(feat, False)
        #         flags.append(prediction.detach().cpu().numpy()[0])
        #
        #         if False:
        #             p = query_points[x_start:x_start + ps, y_start:y_start + ps, z_start:z_start + ps, :].reshape(
        #                 -1, 3)
        #             udf = feat[0][0][0, 0, 0].detach().cpu().numpy().reshape(-1, 1)
        #             gradients = feat[0][0][0, 0, 1:].detach().cpu().numpy().reshape(3, -1).transpose(1, 0)
        #             flag = sigmoid(prediction[0, 0, 0].detach().cpu().numpy().reshape(-1)) > threshold
        #
        #             pp = p + udf * gradients
        #
        #             export_point_cloud("surface.ply", pp)
        #             export_point_cloud("query.ply", p)
        #             export_point_cloud("pred.ply", p[flag])
        #             pass
        #
        #         z_start += ps_2
        #         if z_start + ps > res:
        #             z_start = 0
        #             y_start += ps_2
        #         if y_start + ps > res:
        #             y_start = 0
        #             x_start += ps_2
        #         if x_start + ps > res:
        #             break
        #
        #     flags = np.concatenate(flags, axis=0)
        #
        #     final_flags = np.zeros((res, res, res), dtype=np.float32)
        #     for i in range(flags.shape[0]):
        #         x = i // num_patch_dim // num_patch_dim
        #         y = i // num_patch_dim % num_patch_dim
        #         z = i % num_patch_dim
        #
        #         if x == 0 or x == num_patch_dim - 1 or y == 0 or y == num_patch_dim - 1 or z == 0 or z == num_patch_dim - 1:
        #             final_flags[x * ps_2:x * ps_2 + ps, y * ps_2:y * ps_2 + ps, z * ps_2:z * ps_2 + ps] = flags[
        #                 i, 0]
        #         else:
        #             final_flags[
        #             x * ps_2 + ps_4:x * ps_2 + ps_4 + ps_2,
        #             y * ps_2 + ps_4:y * ps_2 + ps_4 + ps_2,
        #             z * ps_2 + ps_4:z * ps_2 + ps_4 + ps_2] = flags[
        #                                                       i, 0,
        #                                                       ps_4:ps_2 + ps_4,
        #                                                       ps_4:ps_2 + ps_4,
        #                                                       ps_4:ps_2 + ps_4]
        #
        #     final_flags = final_flags
        final_flags = torch.sigmoid(predictions) > threshold

        precision = (final_flags & gt_flags[8:-8,8:-8,8:-8]).sum() / final_flags.sum()
        recall = (final_flags & gt_flags[8:-8,8:-8,8:-8]).sum() / gt_flags[8:-8,8:-8,8:-8].sum()
        f1 = 2 * precision * recall / (precision + recall)
        precisions.append(precision.cpu().numpy())
        recalls.append(recall.cpu().numpy())
        f1s.append(f1.cpu().numpy())

        final_flags = torch.nn.functional.pad(final_flags, (8,8,8,8,8,8), mode="constant", value=0).cpu().numpy()

        final_features = mesh_udf
        valid_points = query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.4))]
        # valid_points = query_points[np.logical_and(gt_flags, (final_features[..., 0] < 0.4))]

        predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
        gradients_and_udf = final_features

        export_point_cloud(str(output_dir / (prefix+".ply")), valid_points)
        np.save(str(output_dir / (prefix+"_feat")), gradients_and_udf)
        np.save(str(output_dir / (prefix+"_pred")), predicted_labels)
        # print("Done")
        bar.update(1)
    print("Precision: {:.4f}".format(np.nanmean(precisions)))
    print("Recall: {:.4f}".format(np.nanmean(recalls)))
    print("F1: {:.4f}".format(np.nanmean(f1s)))
    print("NAN: {:.4f}/{:.4f}".format(np.isnan(f1s).sum(), len(f1s)))


if __name__ == '__main__':
    main()
