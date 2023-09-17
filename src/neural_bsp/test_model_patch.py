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
        v_cfg["model"]
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)

    test_list = v_cfg["dataset"]["test_list"]
    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    ps = 32  # Patch size
    warp = ps // 4

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
            data = dataset[idx]
            feat = [torch.from_numpy(data[0]).cuda(), None]

            prediction = model(feat, False)
            features.append(data[0])
            flags.append(prediction.detach().cpu().numpy())

        features = np.concatenate(features, axis=0)
        flags = np.concatenate(flags, axis=0)

        predicted_labels = (sigmoid(flags[:, 0]) > threshold).astype(np.ubyte)

        total_labels = np.zeros((res, res, res), dtype=np.ubyte)
        total_features = np.zeros((4, res, res, res), dtype=np.float32)
        for idx in range(predicted_labels.shape[0]):
            x,y,z = dataset.patch_list[idx]
            x_start = warp
            x_end = ps - warp
            y_start = warp
            y_end = ps - warp
            z_start = warp
            z_end = ps - warp
            if x==0:
                x_start = 0
            elif x == res - ps:
                x_end = ps
            if y==0:
                y_start = 0
            elif y == res - ps:
                y_end = ps
            if z==0:
                z_start = 0
            elif z == res - ps:
                z_end = ps
            total_labels[x+x_start:x+x_end, y+y_start:y+y_end, z+z_start:z+z_end] = \
                predicted_labels[idx][x_start:x_end, y_start:y_end, z_start:z_end]
            total_features[:, x+x_start:x+x_end, y+y_start:y+y_end, z+z_start:z+z_end] = \
                features[idx][:, x_start:x_end, y_start:y_end, z_start:z_end]

        predicted_labels = total_labels
        features = np.transpose(
            total_features,(1,2,3,0))
        features[:, :, :, 3] /= np.pi

        # Save
        np.save(os.path.join(output_root, "{}_feat.npy".format(name)), features)
        np.save(os.path.join(output_root, "{}_pred.npy".format(name)), predicted_labels)


if __name__ == '__main__':
    main()
