import cv2
import torch
from torch import nn
import numpy as np
from easydict import EasyDict as edict

from shared.img_torch_tools import get_img_from_tensor
from src.img_pair_alignment.original_warp import get_normalized_pixel_grid_crop, warp_grid, warp_corners


class NeuralImageFunction(nn.Module):

    def __init__(self, v_img_height, v_img_width, v_img_crop_size, v_num_sample,
                 v_pos_encoding=8, v_hidden_dim=256):
        super().__init__()
        input_2D_dim = 2 + 4 * v_pos_encoding if v_pos_encoding != -1 else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList()
        self.mlp.append(torch.nn.Linear(input_2D_dim, v_hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(v_hidden_dim, v_hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(v_hidden_dim, v_hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(v_hidden_dim, v_hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(v_hidden_dim, 3))

        scale = np.sqrt(input_2D_dim / 2.)
        self.mlp[0].weight.data *= scale
        self.mlp[0].bias.data *= scale

        self.raw_img_height = v_img_height
        self.raw_img_width = v_img_width
        self.img_crop_size = v_img_crop_size
        self.num_sample = v_num_sample
        self.num_position_encoding = v_pos_encoding
        self.barf_c2f = None
        self.barf_c2f = [0,0.4]

        self.warp_param = torch.nn.Embedding(v_num_sample, 8)
        torch.nn.init.zeros_(self.warp_param.weight)

        self.mse_loss = torch.nn.MSELoss()
        self.progress = 0
        self.max_progress = 5000


    def forward(self, v_img: torch.Tensor):
        assert self.num_sample == v_img.shape[1]
        v_img = v_img[0]
        xy_grid = get_normalized_pixel_grid_crop(self.raw_img_height, self.raw_img_width, self.img_crop_size, self.num_sample, v_img.device)
        xy_grid_warped = warp_grid(xy_grid,self.warp_param.weight)

        # render images
        if self.num_position_encoding != -1:
            points_enc = self.positional_encoding(xy_grid_warped,L=self.num_position_encoding)
            points_enc = torch.cat([xy_grid_warped,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = xy_grid_warped

        feat = points_enc
        # extract implicit features
        for item in self.mlp:
            feat = item(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        rgb_warped_map = rgb.view(v_img.shape[0], v_img.shape[2], v_img.shape[3], 3).permute(0,3,1,2) # [B,3,H,W]
        return rgb_warped_map, self.warp_param.weight

    def loss(self,v_prediction, v_gt, mode=None):
        loss = self.mse_loss(v_prediction, v_gt[0])
        self.progress+=1
        self.warp_param.weight.data[0] = 0
        return loss

    def positional_encoding(self,input, L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if self.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = self.barf_c2f
            alpha = ((self.progress / self.max_progress)-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=input.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc

    def visualize(self, v_prediction, v_gt, v_raw_img, v_step: int):
        v_prediction = v_prediction.detach()
        v_gt = v_gt.detach()
        v_raw_img = v_raw_img.detach()
        with torch.no_grad():
            corners = warp_corners(self.raw_img_height, self.raw_img_width, self.img_crop_size,
                                   v_gt.shape[1], self.warp_param.weight)
            corners[..., 0] = (corners[..., 0] + 1) / 2 * self.raw_img_width - 0.5
            corners[..., 1] = (corners[..., 1] + 1) / 2 * self.raw_img_height - 0.5
            corners = corners.cpu().numpy().astype(np.int)

            concat_imgs=[]
            for i_sample in range(v_gt.shape[1]):  # num_sample
                img_gt = get_img_from_tensor(v_gt[0, i_sample])
                img_prediction = get_img_from_tensor(v_prediction[i_sample])
                raw_img_copy = get_img_from_tensor(v_raw_img[0]).copy()
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
                img_prediction = cv2.cvtColor(img_prediction , cv2.COLOR_RGB2BGR)
                raw_img_copy = cv2.cvtColor(raw_img_copy, cv2.COLOR_RGB2BGR)

                for i_line in range(4):
                    cv2.line(raw_img_copy,
                             (corners[i_sample, i_line, 0], corners[i_sample, i_line, 1]),
                             ((corners[i_sample, (i_line + 1) % 4, 0], corners[i_sample, (i_line + 1) % 4, 1])),
                             (0, 0, 255), 1, cv2.LINE_AA)
                img_gt_pad = np.zeros_like(raw_img_copy)
                img_prediction_pad = np.zeros_like(raw_img_copy)
                img_gt_pad[:img_gt.shape[0], :img_gt.shape[1]] = img_gt
                img_prediction_pad[:img_prediction.shape[0], :img_prediction.shape[1]] = img_prediction
                concat_img = cv2.vconcat([raw_img_copy, img_gt_pad, img_prediction_pad])
                concat_imgs.append(concat_img)
            out_img = cv2.hconcat(concat_imgs)
            cv2.imwrite("output/img_pair_alignment/{}.jpg".format(v_step), out_img)
