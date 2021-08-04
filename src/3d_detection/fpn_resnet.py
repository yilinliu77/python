from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.model_zoo as model_zoo
from visualDet3D.networks.utils.registry import BACKBONE_DICT


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN_Resnet(nn.Module):
    planes = [64, 128, 256, 512]

    def __init__(self, block: Union[BasicBlock, Bottleneck],
                 layers: Tuple[int, ...],
                 num_stages: int = 4,
                 strides: Tuple[int, ...] = (1, 2, 2, 2),
                 dilations: Tuple[int, ...] = (1, 1, 1, 1),
                 out_indices: Tuple[int, ...] = (-1, 0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = True,
                 ):
        self.inplanes = 64
        super(FPN_Resnet, self).__init__()

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        assert max(out_indices) < num_stages

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        #self.smooth0 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # P6 & P7
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)





        # Bottom-up layers
        for i in range(num_stages):
            setattr(self, f"layer{i+1}", self._make_layer(block, self.planes[i], layers[i], stride=self.strides[i], dilation=self.dilations[i]))
        #self.layer1_ = self._make_layer(block,  64, layers[0])
        #self.layer2_ = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3_ = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4_ = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.train()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def train(self, mode=True):
        super(FPN_Resnet, self).train(mode)

        if mode:
            self.freeze_stages()
            if self.norm_eval:
                self.freeze_bn()

    def freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

            for param in self.bn1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):  # Will freeze both batchnorm and sync batchnorm
                layer.eval()

    def forward(self, img_batch):
        # Bottom-up

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        #print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        #print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        #print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        #print(f'c5:{c5.shape}')

        # P6 & P7
        p6 = self.conv6(c5)
        p7 = self.conv7(self.relu(p6))

        # Top-down
        p5 = self.toplayer(c5)
        #print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        #print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        #p5 = self.smooth0(p5)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        out = [p3, p4, p5, p6, p7]

        return out

def fpn_resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FPN_Resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def fpn_resnet(depth, **kwargs):
    if depth == 101:
        model = fpn_resnet101(**kwargs)

    return model

"""
with torch.no_grad():
    model = fpn_resnet101(False).cuda()
    model.eval()
    image = torch.rand(8, 3, 288, 1280).cuda()
    output = model(image)
print(output)

torch.cuda.empty_cache()
"""