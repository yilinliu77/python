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
    log_dir = hydra_cfg['runtime']['output_dir']
    output_root = os.path.join(log_dir, v_cfg["trainer"]["output"])

    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    num_features = v_cfg["model"]["channels"]
    ps = 32
    ps_2 = ps//2
    ps_4 = ps//4
    num_patch_dim = res // ps * 2 - 1
    query_points = generate_coords(res)

    # Load model and dataset
    tasks = [os.path.join(v_cfg["dataset"]["root"],item) for item in os.listdir(v_cfg["dataset"]["root"])]
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

    tasks = tasks[:10]
    # Start inference
    bar = tqdm(total=len(tasks))
    for task in tasks:
        prefix = task.stem.split("_")[0]
        bar.set_description(prefix)
        # print("Start testing {}".format(prefix))
        for type in ["mesh_udf", "pc_udf"]:
            check_dir(task / type)
            mesh_udf = np.load(str(task/"{}.npy".format(type))).reshape(res,res,res,-1)

            flags = []
            x_start = y_start = z_start = 0
            while True:
                feat = mesh_udf[x_start:x_start+ps, y_start:y_start+ps, z_start:z_start+ps, :num_features]
                feat = torch.from_numpy(feat).cuda().permute(3,0,1,2).unsqueeze(0).unsqueeze(0)
                feat = [(feat, torch.zeros(feat.shape[0])), None]

                prediction = model(feat, False)
                flags.append(prediction.detach().cpu().numpy()[0])
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

            final_flags = sigmoid(final_flags) > threshold
            final_features = mesh_udf
            valid_points = query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.2))]

            predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
            gradients_and_udf = final_features

            export_point_cloud(str(task / type/"pred_points.ply"), valid_points)
            np.save(str(task / type/"{}_feat".format(prefix)), gradients_and_udf)
            np.save(str(task / type/"{}_pred".format(prefix)), predicted_labels)
        # print("Done")
        bar.update(1)


if __name__ == '__main__':
    main()
