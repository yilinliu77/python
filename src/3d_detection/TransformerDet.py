import torch
import torchvision
from torch import nn
import sys, os

sys.path.append(os.path.join(__file__, "../../../thirdparty/Swin-Transformer/models"))
from swin_transformer_from_mmdetection import SwinTransformer


# from swin_transformer import SwinTransformer

class Swin_Transformer(nn.Module):
    def __init__(self, v_checkpoint_path=False):
        super(Swin_Transformer, self).__init__()
        self.model = SwinTransformer(pretrain_img_size=224)

    def forward(self, v_input):
        x = self.model(v_input)
        return x


class Swin_Transformer_FPN(nn.Module):
    def __init__(self, v_checkpoint_path=False):
        super(Swin_Transformer_FPN, self).__init__()
        self.swin_transformer = SwinTransformer(pretrain_img_size=224)
        # self.swin_transformer = SwinTransformer(pretrain_img_size=224,
        #                                         embed_dim=128,
        #                                         depths=[2, 2, 18, 2],
        #                                         num_heads=[4, 8, 16, 32],
        #                                         drop_path_rate=0.3, )

        self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork([96, 192, 384, 768], 512)
        # self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork([128, 256, 512, 1024], 512)

    def forward(self, v_input):
        result = self.swin_transformer(v_input)
        features = {
            "feat0": result[0],
            "feat1": result[1],
            "feat2": result[2],
            "feat3": result[3],
        }
        result = self.fpn(features)
        return [result["feat0"], result["feat1"], result["feat2"], result["feat3"]]


class TransformerDet(nn.Module):
    def __init__(self):
        super(TransformerDet, self).__init__()


if __name__ == '__main__':
    model = Swin_Transformer()
    model_fpn = Swin_Transformer_FPN()
    result = model(torch.randn((1, 3, 1280, 288), dtype=torch.float32))
    result_fpn = model_fpn(torch.randn((1, 3, 1280, 288), dtype=torch.float32))

    pass
