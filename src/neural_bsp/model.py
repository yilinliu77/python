from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchvision.ops import sigmoid_focal_loss

def focal_loss(v_predictions, labels, v_alpha=0.75):
    loss = sigmoid_focal_loss(v_predictions, labels,
                              alpha=v_alpha,
                              reduction="mean"
                              )
    return loss

def BCE_loss(v_predictions, labels, v_alpha=0.75):
    loss = nn.functional.binary_cross_entropy_with_logits(v_predictions, labels,
                                                          reduction="mean"
                                                          )
    return loss

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, with_bn=True):
        super(conv_block, self).__init__()
        if with_bn:
            self.conv1 = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True),

            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # down-sample
        g1 = self.W_g(g)
        # up-sample
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # weight matrix
        psi = self.psi(psi)
        # weighted x
        return x * psi


class AttU_Net_3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, v_depth=4):
        super(AttU_Net_3D, self).__init__()
        self.depths = v_depth
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = nn.ModuleList()
        self.up = nn.ModuleList()
        self.att = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        base_channel = 16
        self.conv1 = nn.Sequential(
            conv_block(ch_in=img_ch, ch_out=base_channel),
            nn.MaxPool3d(kernel_size=4, stride=4),
            conv_block(ch_in=base_channel, ch_out=base_channel),
        )
        cur_channel = base_channel
        for i in range(v_depth):
            self.conv.append(conv_block(ch_in=cur_channel, ch_out=cur_channel * 2))

            self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
            self.up_conv.append(conv_block(ch_in=cur_channel * 2, ch_out=cur_channel))
            self.att.append(Attention_block(F_g=cur_channel, F_l=cur_channel, F_int=cur_channel // 2))

            cur_channel = cur_channel * 2

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 1, -1, -1):
            item = self.up[i](up_x[-1])
            x_i = self.att[i](g=item, x=x[i])
            item = torch.cat((item, x_i), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])
        return d1


class U_Net_3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, v_depth=5, v_pool_first=True, base_channel=16, with_bn=True):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.depths = v_depth

        self.conv = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        if v_pool_first:
            self.conv1 = nn.Sequential(
                conv_block(ch_in=img_ch, ch_out=base_channel, with_bn=with_bn),
                nn.MaxPool3d(kernel_size=4, stride=4),
                conv_block(ch_in=base_channel, ch_out=base_channel, with_bn=with_bn),
            )
        elif with_bn:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(base_channel),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        cur_channel = base_channel
        for i in range(v_depth):
            self.conv.append(conv_block(ch_in=cur_channel, ch_out=cur_channel * 2, with_bn=with_bn))

            self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
            self.up_conv.append(conv_block(ch_in=cur_channel * 2, ch_out=cur_channel, with_bn=with_bn))

            cur_channel = cur_channel * 2

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, v_input):
        # encoding path
        x1 = self.conv1(v_input)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 1, -1, -1):
            item = self.up[i](up_x[-1])
            item = torch.cat((item, x[i]), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class Base_model(nn.Module):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5):
        super(Base_model, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1)
        self.loss_func = globals()[v_loss_type]
        self.loss_alpha = v_alpha
        # self.encoder = AttU_Net_3D(img_ch=4, output_ch=1)

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = self.loss_func(v_predictions, labels, self.loss_alpha)
        return loss


class Atten_model(nn.Module):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5):
        super(Atten_model, self).__init__(v_phase, v_loss_type, v_alpha)
        self.phase = v_phase
        self.encoder = AttU_Net_3D(img_ch=3, output_ch=1)

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction


class Base_patch_model_focal(Base_model):
    def __init__(self, v_phase=0):
        super(Base_patch_model_focal, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1, v_pool_first=False, v_depth=4)


class Base_patch_model_BCE_deeper(Base_model):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5):
        super(Base_patch_model_BCE_deeper, self).__init__(v_phase, v_loss_type, v_alpha)
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1, v_pool_first=False, v_depth=5, base_channel=32)


class Base_patch_model_BCE_deeper_wo_bn(Base_model):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5):
        super(Base_patch_model_BCE_deeper_wo_bn, self).__init__(v_phase, v_loss_type, v_alpha)
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1, v_pool_first=False, v_depth=5, base_channel=32, with_bn=False)


class Base_patch_model_BCE_dir(Base_model):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5):
        super(Base_patch_model_BCE_dir, self).__init__(v_phase, v_loss_type, v_alpha)
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=4, output_ch=1, v_pool_first=False, v_depth=4, base_channel=16)