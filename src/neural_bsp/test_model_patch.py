import importlib
import os.path
from pathlib import Path

import hydra
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from shared.common_utils import export_point_cloud, sigmoid
from src.neural_bsp.abc_hdf5_dataset import ABC_dataset_test_mesh

@hydra.main(config_name="test_model_patch.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    output_root = os.path.join(log_dir, v_cfg["trainer"]["output"])

    mod = importlib.import_module('src.neural_bsp.model')
    model = getattr(mod, v_cfg["model"]["model_name"])(
        v_cfg["model"]["phase"],
        v_cfg["model"]["loss"],
        v_cfg["model"]["loss_alpha"],
        v_cfg["model"]["v_input_channel"],
        v_cfg["model"]["v_depth"],
        v_cfg["model"]["v_base_channel"],
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)

    test_list = v_cfg["dataset"]["test_list"]
    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    ps = 32  # Patch size
    npd = res // ps  # Number of patches per dimension

    for model_name in test_list:
        print("Start testing {}".format(model_name))
        name = Path(model_name).stem
        dataset = ABC_dataset_test_mesh(
            model_name,
            v_cfg["trainer"]["batch_size"],
            v_cfg["dataset"]["v_output_features"],
            res
        )

        features = []
        flags = []
        for idx in tqdm(range(len(dataset))):
            data = [torch.from_numpy(item).cuda() for item in dataset[idx]]

            prediction = model(data, False)
            features.append(data[0].detach().cpu().numpy())
            flags.append(prediction.detach().cpu().numpy())

        features = np.concatenate(features, axis=0)
        flags = np.concatenate(flags, axis=0)

        predicted_labels = (sigmoid(flags[:, 0]) > threshold).astype(np.ubyte)
        predicted_labels = np.transpose(
            predicted_labels.reshape((npd, npd, npd, ps, ps, ps)),
            (0, 3, 1, 4, 2, 5)).reshape((res, res, res))
        features = np.transpose(
            features.reshape((npd, npd, npd, -1, ps, ps, ps)),
            (0, 4, 1, 5, 2, 6, 3)).reshape((res, res, res, -1))
        features[:, :, :, 3] /= np.pi

        # Save
        np.save(os.path.join(output_root, "{}_feat.npy".format(name)), features)
        np.save(os.path.join(output_root, "{}_pred.npy".format(name)), predicted_labels)


if __name__ == '__main__':
    main()
