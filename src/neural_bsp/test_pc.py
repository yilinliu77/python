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
from src.neural_bsp.abc_hdf5_dataset import ABC_test_mesh, generate_coords


@hydra.main(config_name="test_pc.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    # Predefined variables
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    output_root = os.path.join(log_dir, v_cfg["trainer"]["output"])

    test_list = v_cfg["dataset"]["test_list"]  # List of models to test
    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    query_points = generate_coords(res)

    # Load model and dataset
    dataset_module = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
    dataset_name = getattr(dataset_module, v_cfg["dataset"]["dataset_name"])

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

    # Start inference
    for model_name in test_list:
        print("Start testing {}".format(model_name))
        name = Path(model_name).stem

        dataset = dataset_name(
            model_name,
            v_cfg["trainer"]["batch_size"],
            res,
            output_root,
        )

        data = dataset[0]
        feat = [(torch.from_numpy(data[0]).cuda().unsqueeze(0), torch.zeros(data[0].shape[0])), None]

        prediction = model(feat, False).cpu().numpy()
        final_flags = sigmoid(prediction.reshape(res,res,res)) > threshold

        final_features = dataset.feat_data
        valid_points = query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.2))]

        export_point_cloud(os.path.join(output_root, "{}_pred_points.ply".format(name)),
                           valid_points)

        predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
        gradients_and_udf = final_features

        np.save(os.path.join(output_root, "{}_feat.npy".format(name)), gradients_and_udf)
        np.save(os.path.join(output_root, "{}_pred.npy".format(name)), predicted_labels)
        export_point_cloud(os.path.join(output_root, "{}_points.ply".format(name)), dataset.poisson_points)
        print("Done")


if __name__ == '__main__':
    main()
