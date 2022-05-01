import time

import hydra
import torch
from flask import Flask, request
from omegaconf import OmegaConf, DictConfig
from plyfile import PlyData
from pytorch_lightning import seed_everything

from src.regress_reconstructability_hyper_parameters.model import Uncertainty_Modeling_v2, \
    Uncertainty_Modeling_w_pointnet, Uncertainty_Modeling_wo_pointnet, Uncertainty_Modeling_wo_pointnet5, \
    Uncertainty_Modeling_wo_pointnet7, Uncertainty_Modeling_wo_pointnet14, Correlation_l2_error_net
from src.regress_reconstructability_hyper_parameters.train import Regress_hyper_parameters
import numpy as np

app = Flask(__name__)

model = None


@app.route("/", methods=["GET", "POST"])
def main():
    a = request
    pass


@hydra.main(config_name="test.yaml", config_path="../../configs/regress_hyper_parameters/")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    v_cfg["model"]["model_name"] = "Correlation_l2_error_net"
    # v_cfg["model"]["model_name"] = "Spearman_net"
    v_cfg["model"]["sigmoid"] = False
    v_cfg["model"]["spearman_method"] = "kl"
    v_cfg["model"]["error_mean_std"] = [0.05, 0.05, 0.05, 0.05]
    v_cfg["model"]["involve_img"] = True

    global model
    # model = Uncertainty_Modeling_w_pointnet(v_cfg)
    model = Correlation_l2_error_net(v_cfg)
    best_model = torch.load(r"temp/recon_model/Correlation_l2_error_net_p1.ckpt")
    model.load_state_dict(
        {item.split("model.")[1]: best_model["state_dict"][item] for item in best_model["state_dict"]})
    model.eval()
    sm = torch.jit.script(model)
    sm.save("temp/recon_model/model.pt")
    sm.eval()
    sm.cuda()

    views = torch.rand((512,1,100,8),dtype=torch.float32).cuda()
    point_features = torch.rand((512,1,50,32),dtype=torch.float32).cuda()
    point_features_mask = torch.ones((512,1,50),dtype=torch.bool).cuda()
    data_new = {"views": views,
                "point_features": point_features,
                "point_features_mask": point_features_mask
                }
    model.cuda()
    with torch.no_grad():
        running_time = 0
        results2 = sm.forward(data_new)
        for i in range(10):
            cur = time.time()
            results2 = sm.forward(data_new)
            running_time += time.time() - cur
        print(running_time / 10)
        print(views.shape)
        results1 = model.forward(data_new)
        result_python_flatten = np.zeros(num_points)
        result_cpp_flatten = np.zeros(num_points)
        result_cpp_flatten[points[:, :, 3:4].int().cpu().numpy()] = results2[:, :, 0:1].cpu().numpy()
        result_python_flatten[points[:, :, 3:4].int().cpu().numpy()] = results1[:, :, 0:1].cpu().numpy()
        assert np.abs(result_python_flatten - result_cpp_flatten).sum() < 1e-6

    with open(r"D:\Projects\Reconstructability\PathPlanning\test_xuexiao_sparse\accuracy_projected.ply", "rb") as f:
        plydata = PlyData.read(f)
        gt_error = plydata['vertex']['sum_error'].copy() / plydata['vertex']['num'].copy()
        visible_index = result_python_flatten != 0
        l2_loss = np.mean((gt_error[visible_index] - result_python_flatten[visible_index]) ** 2)

    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        for item in list(state_dict.keys()):
            if "point_feature_extractor" in item:
                state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)
    app.run()


if __name__ == '__main__':
    main()
