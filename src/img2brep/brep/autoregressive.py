from torch import nn

from src.img2brep.brep.model import AutoEncoder


class AutoregressiveModel(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.autoencoder = AutoEncoder(self.hydra_conf["model"])
        model_path = v_conf["checkpoint_autoencoder"]

        # Load model
        pass

    def forward(self, v_data):

        pass

    def loss(self, v_prediction, v_data):

        pass