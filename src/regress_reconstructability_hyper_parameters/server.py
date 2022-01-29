import hydra
import torch
from flask import Flask, request
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything

from src.regress_reconstructability_hyper_parameters.model import Uncertainty_Modeling_v2, \
    Uncertainty_Modeling_w_pointnet
from src.regress_reconstructability_hyper_parameters.train import Regress_hyper_parameters

app = Flask(__name__)

model = None

@app.route("/",methods=["GET","POST"])
def main():
    a=request
    pass

@hydra.main(config_name="test.yaml")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    global model
    model = Uncertainty_Modeling_w_pointnet(v_cfg)
    best_model = torch.load(r"temp/continuous_error.ckpt")
    model.load_state_dict({item.split("model.")[1]:best_model["state_dict"][item] for item in best_model["state_dict"]})
    model.eval()
    sm = torch.jit.script(model)
    sm.save("temp/model.pt")
    sm.eval()
    sm.cuda()

    views = torch.load(r"C:\repo\C\build\src\sig22_reconstructability\optimize_trajectory\views.pt").cuda()
    points = torch.load(r"C:\repo\C\build\src\sig22_reconstructability\optimize_trajectory\points.pt").cuda()
    point_attribute = torch.load(r"C:\repo\C\build\src\sig22_reconstructability\optimize_trajectory\point_attributes.pt").cuda()
    data_new = {"views": views,
                "points": points,
                "point_attribute":point_attribute}
    model.cuda()
    with torch.no_grad():
        results1 = model.forward(data_new)
        results2 = sm.forward(data_new)
        assert torch.abs(results1[:,:,0]-results2[:,:,0]).sum() < 1e-6

    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        for item in list(state_dict.keys()):
            if "point_feature_extractor" in item:
                state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)
    app.run()


if __name__ == '__main__':
    main()