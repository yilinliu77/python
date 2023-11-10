import importlib
import os.path
from pathlib import Path

import hydra
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from shared.common_utils import *
from src.neural_bsp.abc_hdf5_dataset import ABC_test_mesh, generate_coords

#
# Test both mesh udf and udc udf in the testset
#

@hydra.main(config_name="test_model.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    # Predefined variables
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_root = Path(v_cfg["dataset"]["root"])

    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    num_features = v_cfg["model"]["channels"]
    ps = 32
    ps_2 = ps//2
    ps_4 = ps//4
    num_patch_dim = res // ps * 2 - 1
    query_points = generate_coords(res)

    check_dir(output_root / "mesh_udf_prediction")
    check_dir(output_root / "pc_udf_prediction")

    # Load model and dataset
    tasks = [os.path.join(v_cfg["dataset"]["root"],"mesh_udf",item) for item in os.listdir(
        os.path.join(v_cfg["dataset"]["root"], "mesh_udf"))]
    tasks = sorted(tasks)
    tasks = [Path(item) for item in tasks]

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

    tasks = tasks[9:10]
    # Start inference
    bar = tqdm(total=len(tasks))
    for task in tasks:
        prefix = task.stem
        bar.set_description(prefix)
        # print("Start testing {}".format(prefix))
        for type in ["mesh_udf", "pc_udf"]:
        # for type in ["pc_udf"]:
            mesh_udf = np.load(
                os.path.join(v_cfg["dataset"]["root"], type, "{}.npy".format(prefix))
            ).reshape(res,res,res,-1)

            if True:
                flags = -np.ones((res, res, res), dtype=np.float32) * 999
                x_start = y_start = z_start = 0
                while True:
                    feat = mesh_udf[
                           x_start:x_start + ps,
                           y_start:y_start + ps,
                           z_start:z_start + ps,
                           :num_features
                           ]
                    feat = torch.from_numpy(feat).cuda().permute(3, 0, 1, 2).unsqueeze(0).unsqueeze(0)
                    feat = [(feat, torch.zeros(feat.shape[0])), None]
                    prediction = model(feat, False).reshape(ps,ps,ps)

                    flags[
                        x_start:x_start + ps,
                        y_start:y_start + ps,
                        z_start:z_start + ps
                    ] = np.maximum(prediction.cpu().numpy(), flags[
                        x_start:x_start + ps,
                        y_start:y_start + ps,
                        z_start:z_start + ps
                    ])

                    z_start += ps_2
                    if z_start + ps > res:
                        z_start = 0
                        y_start += ps_2
                    if y_start + ps > res:
                        y_start = 0
                        x_start += ps_2
                    if x_start + ps > res:
                        break

                final_flags = flags
            else:
                flags = []
                x_start = y_start = z_start = 0
                while True:
                    feat = mesh_udf[x_start:x_start+ps, y_start:y_start+ps, z_start:z_start+ps, :num_features]
                    feat = torch.from_numpy(feat).cuda().permute(3,0,1,2).unsqueeze(0).unsqueeze(0)
                    feat = [(feat, torch.zeros(feat.shape[0])), None]

                    prediction = model(feat, False)
                    flags.append(prediction.detach().cpu().numpy()[0])

                    if False:
                        p = query_points[x_start:x_start+ps, y_start:y_start+ps, z_start:z_start+ps, :].reshape(-1,3)
                        udf = feat[0][0][0,0,0].detach().cpu().numpy().reshape(-1,1)
                        gradients = feat[0][0][0,0,1:].detach().cpu().numpy().reshape(3,-1).transpose(1,0)
                        flag = sigmoid(prediction[0,0,0].detach().cpu().numpy().reshape(-1))>threshold

                        pp = p + udf * gradients

                        export_point_cloud("surface.ply", pp)
                        export_point_cloud("query.ply", p)
                        export_point_cloud("pred.ply", p[flag])
                        pass

                    z_start += ps_2
                    if z_start + ps > res:
                        z_start = 0
                        y_start += ps_2
                    if y_start + ps > res:
                        y_start = 0
                        x_start += ps_2
                    if x_start + ps > res:
                        break

                flags = np.concatenate(flags, axis=0)

                final_flags = np.zeros((res, res, res), dtype=np.float32)
                for i in range(flags.shape[0]):
                    x = i // num_patch_dim // num_patch_dim
                    y = i // num_patch_dim % num_patch_dim
                    z = i % num_patch_dim

                    if x == 0 or x == num_patch_dim-1 or y == 0 or y == num_patch_dim-1 or z == 0 or z == num_patch_dim-1:
                        final_flags[x * ps_2:x * ps_2 + ps, y * ps_2:y * ps_2 + ps, z * ps_2:z * ps_2 + ps] = flags[i, 0]
                    else:
                        final_flags[
                        x * ps_2 + ps_4:x * ps_2 + ps_4 + ps_2,
                        y * ps_2 + ps_4:y * ps_2 + ps_4 + ps_2,
                        z * ps_2 + ps_4:z * ps_2 + ps_4 + ps_2] = flags[
                                                                  i, 0,
                                                                  ps_4:ps_2 + ps_4,
                                                                  ps_4:ps_2 + ps_4,
                                                                  ps_4:ps_2 + ps_4]

                final_flags = final_flags
            final_flags = sigmoid(final_flags) > threshold
            final_features = mesh_udf
            valid_points = query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.4))]

            predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
            gradients_and_udf = final_features

            export_point_cloud(str(output_root / (type+"_prediction") /"{}_pred_points.ply".format(prefix)), valid_points)
            np.save(str(output_root / (type+"_prediction") /"{}_feat".format(prefix)), gradients_and_udf)
            np.save(str(output_root / (type+"_prediction") /"{}_pred".format(prefix)), predicted_labels)
        # print("Done")
        bar.update(1)


if __name__ == '__main__':
    main()
