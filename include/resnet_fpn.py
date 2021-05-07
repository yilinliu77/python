import torch,torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


if __name__ == '__main__':
    resnet_fpn_backbone(
        'resnet50',
        pretrained=True,
        trainable_layers=5,
    )

