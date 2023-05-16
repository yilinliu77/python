import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from src.neural_recon.optimize_segment import sample_img


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        y = y.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def gradient(pred):
    D_dy = (pred[:, 0:-2, :, :] - pred[:, 2:, :, :])/2
    D_dx = (pred[:, :, 0:-2, :] - pred[:, :, 2:, :])/2
    return D_dx, D_dy


def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    # print('depth: {} img: {}'.format(depth.shape, img.shape))
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    # print('depth_dx: {} weights_x: {}'.format(depth_dx.shape, weights_x.shape))
    # print('depth_dy: {} weights_y: {}'.format(depth_dy.shape, weights_y.shape))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        photo_loss = F.smooth_l1_loss(warped * mask, ref * mask, reduction='none')
        photo_loss = torch.sum(photo_loss, dim=[1,2,3]) / torch.sum(mask.to(torch.int32), dim=[1,2,3])
        return photo_loss
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped * mask, ref * mask, reduction='none')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='none')[:,1:-1:,:]\
                        + F.smooth_l1_loss(warped_dy, ref_dy, reduction='none')[:,:,1:-1,:]
        photo_loss = torch.sum(photo_loss, dim=[1,2,3]) / torch.sum(mask.to(torch.int32), dim=[1,2,3])
        grad_loss = torch.sum(grad_loss, dim=[1,2,3]) / torch.sum(mask[:,1:-1,1:-1,:].to(torch.int32), dim=[1,2,3])
        return (1 - alpha) * photo_loss + alpha * grad_loss


def loss1(sample_imgs1, sample_imgs2, v_points1, v_points2, v_num_points):
    device = sample_imgs1.device
    times=[0 for _ in range(10)]
    cur_time = time.time()
    start_length = 0
    maxx = torch.max(v_num_points)//20
    num_imgs = v_num_points.shape[0]

    m = torch.arange(maxx*20,device=device).repeat(v_num_points.shape[0]).reshape(v_num_points.shape[0], maxx*20)
    imgs_mask = m < v_num_points[:,None]
    imgs_mask = imgs_mask.reshape(v_num_points.shape[0],maxx, 20, 1)

    imgs1 = torch.zeros((num_imgs, maxx, 20, 1), device=sample_imgs1.device, dtype=sample_imgs1.dtype)
    imgs2 = torch.zeros((num_imgs, maxx, 20, 1), device=sample_imgs1.device, dtype=sample_imgs1.dtype)
    imgs1[imgs_mask] = sample_imgs1[:,0]
    imgs2[imgs_mask] = sample_imgs2[:,0]

    imgs1 = torch.ones((num_imgs, maxx, 20, 1), device=sample_imgs1.device, dtype=sample_imgs1.dtype)
    imgs2 = torch.ones((num_imgs, maxx, 20, 1), device=sample_imgs1.device, dtype=sample_imgs1.dtype)
    imgs_mask = torch.zeros((num_imgs, maxx, 20, 1), device=sample_imgs1.device, dtype=torch.bool)
    for idx, length in enumerate(v_num_points):
        img1 = sample_imgs1[start_length:start_length+length].reshape(-1, 20)
        img2 = sample_imgs2[start_length:start_length+length].reshape(-1, 20)
        start_length += length
        imgs1[idx,:img1.shape[0],:,0] *= img1
        imgs2[idx,:img2.shape[0],:,0] *= img2
        imgs_mask[idx,:img2.shape[0],:,0] = True
    imgs1 = imgs1 * imgs_mask.to(torch.int32)
    imgs2 = imgs2 * imgs_mask.to(torch.int32)

    times[0] += time.time() - cur_time
    cur_time = time.time()

    sigma_color = 0.2
    sigma_spatial = 10
    spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
    color_normalization_=1. / (2. * sigma_color * sigma_color)

    spatial_weights = torch.linspace(-10,10,20,device=v_points1.device,dtype=torch.float32)**2 * spatial_normalization_
    color_weights = ((imgs1-imgs1[:,:,9:10,:])*1)**2 * color_normalization_
    spatial_weights = torch.tile(spatial_weights[None,None,:,None], (num_imgs, maxx, 1, 1))
    bilateral_weight = torch.exp(-spatial_weights-color_weights)
    bilateral_weight = bilateral_weight * imgs_mask.to(torch.int32)
    ref_color_sum = (bilateral_weight * imgs1).sum(dim=[1,2,3]) / bilateral_weight.sum(dim=[1,2,3])
    ref_color_squared_sum = (bilateral_weight * imgs1 * imgs1).sum(dim=[1,2,3]) / bilateral_weight.sum(dim=[1,2,3])

    bilateral_weight_src = bilateral_weight * imgs2
    src_color_sum = bilateral_weight_src.sum(dim=[1,2,3])
    src_color_squared_sum = (bilateral_weight_src * imgs2).sum(dim=[1,2,3])
    src_ref_color_sum = (bilateral_weight_src * imgs1).sum(dim=[1,2,3])
    bilateral_weight_sum = bilateral_weight.sum(dim=[1,2,3])

    inv_bilateral_weight_sum = 1. / bilateral_weight_sum
    src_color_sum *= inv_bilateral_weight_sum
    src_color_squared_sum *= inv_bilateral_weight_sum
    src_ref_color_sum *= inv_bilateral_weight_sum
    ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum
    src_color_var = src_color_squared_sum - src_color_sum * src_color_sum

    src_ref_color_covar = src_ref_color_sum - ref_color_sum * src_color_sum
    src_ref_color_var = torch.sqrt(ref_color_var * src_color_var)
    # bilateral_ncc = torch.clamp_min(torch.clamp_max(1. - src_ref_color_covar / src_ref_color_var, 2), 0)
    bilateral_ncc = 1. - src_ref_color_covar / src_ref_color_var

    times[1] += time.time() - cur_time
    cur_time = time.time()

    # ssim_loss = ssim(
    #     imgs1,
    #     imgs2,
    #     imgs_mask)
    # ssim_loss = ssim_loss * gradient_mask
    # ssim_loss = torch.sum(ssim_loss, dim=[1,2,3]) / torch.sum(gradient_mask, dim=[1,2,3])
    # photo_loss = compute_reconstr_loss(
    #     imgs1,
    #     imgs2,
    #     imgs_mask,
    #     simple=False
    # )
    # # loss = ssim_loss + photo_loss * 12
    # loss = photo_loss
    # return loss
    return bilateral_ncc


def loss2(sample_imgs1, sample_imgs2, v_num_points):
    device = sample_imgs1.device
    times=[0 for _ in range(10)]
    cur_time = time.time()
    start_length = 0
    maxx = torch.max(v_num_points)//20
    num_imgs = sample_imgs2.shape[0]
    num_edges = v_num_points.shape[0]

    m = torch.arange(maxx*20,device=device).repeat(v_num_points.shape[0]).reshape(v_num_points.shape[0], maxx*20)
    imgs_mask = m < v_num_points[:,None]
    imgs_mask = imgs_mask.reshape(v_num_points.shape[0],maxx, 20, 1)

    imgs1 = torch.zeros((num_edges, maxx, 20, 1), device=device, dtype=sample_imgs1.dtype)
    imgs2 = torch.zeros((num_imgs, num_edges, maxx, 20, 1), device=device, dtype=sample_imgs1.dtype)
    imgs1[imgs_mask] = sample_imgs1[:,0]
    imgs2[imgs_mask[None,:].tile([num_imgs,1,1,1,1])] = sample_imgs2.reshape(-1)

    # Check if pixels in the reference image are most black
    avg_pixel_color1 = scatter_mean(sample_imgs1,
                                   torch.arange(v_num_points.shape[0], device=device).repeat_interleave(v_num_points),
                                   dim=0)
    black_area_in_img1 = avg_pixel_color1 < 0.05

    avg_pixel_color = scatter_mean(sample_imgs2,
                 torch.arange(v_num_points.shape[0],device=device).repeat_interleave(v_num_points),
                 dim=1)
    penalize_black_background_loss = (0.1-torch.clamp_max(avg_pixel_color,0.1)) * 10

    times[0] += time.time() - cur_time
    cur_time = time.time()

    sigma_color = 0.2
    sigma_spatial = 10
    spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
    color_normalization_=1. / (2. * sigma_color * sigma_color)

    spatial_weights = torch.linspace(-10,10,20,device=device,dtype=torch.float32)**2 * spatial_normalization_
    color_weights = ((imgs1-imgs1[:,:,9:10,:])*1)**2 * color_normalization_
    spatial_weights = torch.tile(spatial_weights[None,None,:,None], (num_edges, maxx, 1, 1))
    bilateral_weight = torch.exp(-spatial_weights-color_weights)
    bilateral_weight = bilateral_weight * imgs_mask.to(torch.int32)
    ref_color_sum = (bilateral_weight * imgs1).sum(dim=[1,2,3]) / bilateral_weight.sum(dim=[1,2,3])
    ref_color_squared_sum = (bilateral_weight * imgs1 * imgs1).sum(dim=[1,2,3]) / bilateral_weight.sum(dim=[1,2,3])

    bilateral_weight_src = bilateral_weight[None, :] * imgs2
    src_color_sum = bilateral_weight_src.sum(dim=[2,3,4])
    src_color_squared_sum = (bilateral_weight_src * imgs2).sum(dim=[2,3,4])
    src_ref_color_sum = (bilateral_weight_src * imgs1[None, :]).sum(dim=[2,3,4])
    bilateral_weight_sum = bilateral_weight.sum(dim=[1,2,3])

    inv_bilateral_weight_sum = 1. / bilateral_weight_sum
    src_color_sum *= inv_bilateral_weight_sum[None,:]
    src_color_squared_sum *= inv_bilateral_weight_sum[None,:]
    src_ref_color_sum *= inv_bilateral_weight_sum[None,:]
    ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum
    src_color_var = src_color_squared_sum - src_color_sum * src_color_sum

    valid_mask1 = ref_color_var>1e-6
    valid_mask = torch.logical_and(src_color_var>1e-6, valid_mask1[None,:].tile([num_imgs,1]))

    bilateral_ncc = torch.ones_like(src_color_var)
    src_ref_color_covar = src_ref_color_sum - ref_color_sum[None,:] * src_color_sum
    t = ref_color_var[None,:] * src_color_var
    src_ref_color_var = torch.sqrt(t[valid_mask])
    bilateral_ncc[valid_mask] *= (1. - src_ref_color_covar[valid_mask] / src_ref_color_var)
    bilateral_ncc[~valid_mask] *= 5 # Set it to a large value
    times[1] += time.time() - cur_time
    cur_time = time.time()

    # return bilateral_ncc + penalize_black_background_loss[:,:,0], black_area_in_img1
    return bilateral_ncc, black_area_in_img1

# Pure NCC
def loss3(sample_imgs1, sample_imgs2, v_num_points):
    device = sample_imgs1.device
    times=[0 for _ in range(10)]
    cur_time = time.time()
    start_length = 0
    maxx = torch.max(v_num_points)//20
    num_imgs = sample_imgs2.shape[0]
    num_edges = v_num_points.shape[0]

    m = torch.arange(maxx*20,device=device).repeat(v_num_points.shape[0]).reshape(v_num_points.shape[0], maxx*20)
    imgs_mask = m < v_num_points[:,None]
    imgs_mask = imgs_mask.reshape(v_num_points.shape[0],maxx, 20, 1)

    imgs1 = torch.zeros((num_edges, maxx, 20, 1), device=device, dtype=sample_imgs1.dtype)
    imgs2 = torch.zeros((num_imgs, num_edges, maxx, 20, 1), device=device, dtype=sample_imgs1.dtype)
    imgs1[imgs_mask] = sample_imgs1[:,0]
    imgs2[imgs_mask[None,:].tile([num_imgs,1,1,1,1])] = sample_imgs2.reshape(-1)

    # Check if pixels in the reference image are most black
    repeat_index = torch.arange(v_num_points.shape[0],device=device).repeat_interleave(v_num_points)
    avg_pixel_color1 = scatter_mean(sample_imgs1,
                                   repeat_index,
                                   dim=0)
    black_area_in_img1 = avg_pixel_color1 < 0.05

    avg_pixel_color2 = scatter_mean(sample_imgs2,
                 repeat_index,
                 dim=1)

    times[0] += time.time() - cur_time
    cur_time = time.time()

    A_zero_mean = sample_imgs1 - avg_pixel_color1.repeat_interleave(v_num_points,dim=0)
    B_zero_mean = sample_imgs2 - avg_pixel_color2.repeat_interleave(v_num_points,dim=1)

    dot_product = scatter_sum(A_zero_mean * B_zero_mean, repeat_index,dim=1)
    A_norm = torch.sqrt(scatter_sum(A_zero_mean ** 2,repeat_index,dim=0))
    B_norm = torch.sqrt(scatter_sum(B_zero_mean ** 2,repeat_index,dim=1))

    NCC = dot_product / (A_norm * B_norm + 1e-8)
    times[1] += time.time() - cur_time
    cur_time = time.time()

    return NCC, black_area_in_img1

# Pure NCC with spatial weights
def loss4(sample_imgs1, sample_imgs2, v_num_points):
    device = sample_imgs1.device
    times=[0 for _ in range(10)]
    cur_time = time.time()
    start_length = 0
    maxx = torch.max(v_num_points)//20
    num_imgs = sample_imgs2.shape[0]
    num_edges = v_num_points.shape[0]

    # Add weights
    sigma_spatial = 10
    spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
    spatial_weights = torch.linspace(-10,10,20,device=device,dtype=torch.float32)**2 * spatial_normalization_
    spatial_weights = spatial_weights.tile((v_num_points//20).sum())
    spatial_weights = torch.exp(-spatial_weights)

    repeat_index = torch.arange(v_num_points.shape[0],device=device).repeat_interleave(v_num_points)
    spatial_weights_sum = scatter_sum(spatial_weights, repeat_index)
    spatial_weights = spatial_weights / spatial_weights_sum.repeat_interleave(v_num_points)

    avg_pixel_color1 = scatter_mean(sample_imgs1,
                                   repeat_index,
                                   dim=0)
    black_area_in_img1 = avg_pixel_color1 < 0.05

    # sample_imgs1 = sample_imgs1 * spatial_weights[:,None]
    # sample_imgs2 = sample_imgs2 * spatial_weights[None,:,None]

    # Check if pixels in the reference image are most black
    # avg_pixel_color1 = scatter_mean(sample_imgs1,repeat_index,dim=0)
    avg_pixel_color2 = scatter_mean(sample_imgs2,repeat_index,dim=1)

    times[0] += time.time() - cur_time
    cur_time = time.time()

    A_zero_mean = (sample_imgs1 - avg_pixel_color1.repeat_interleave(v_num_points,dim=0)) * spatial_weights[:,None]
    B_zero_mean = (sample_imgs2 - avg_pixel_color2.repeat_interleave(v_num_points,dim=1)) * spatial_weights[None,:,None]

    dot_product = scatter_sum(A_zero_mean * B_zero_mean, repeat_index,dim=1)
    A_norm = torch.sqrt(torch.clamp_min(scatter_sum(A_zero_mean ** 2,repeat_index,dim=0), 1e-6))
    B_norm = torch.sqrt(torch.clamp_min(scatter_sum(B_zero_mean ** 2,repeat_index,dim=1),1e-6))

    NCC = 1 - dot_product / (torch.clamp_min(A_norm * B_norm, 1e-6))
    times[1] += time.time() - cur_time
    cur_time = time.time()

    return NCC, black_area_in_img1

# Pure face NCC
def loss5(sample_imgs1, sample_imgs2, v_num_points):
    device = sample_imgs1.device
    times=[0 for _ in range(10)]
    cur_time = time.time()
    num_face = v_num_points.shape[0]

    repeat_index = torch.arange(num_face,device=device).repeat_interleave(v_num_points)

    avg_pixel_color1 = scatter_mean(sample_imgs1,repeat_index,dim=0)
    avg_pixel_color2 = scatter_mean(sample_imgs2,repeat_index,dim=0)

    times[0] += time.time() - cur_time
    cur_time = time.time()

    A_zero_mean = sample_imgs1 - avg_pixel_color1.repeat_interleave(v_num_points,dim=0)
    B_zero_mean = sample_imgs2 - avg_pixel_color2.repeat_interleave(v_num_points,dim=0)

    dot_product = scatter_sum(A_zero_mean * B_zero_mean, repeat_index,dim=0)
    A_norm = torch.sqrt(torch.clamp_min(scatter_sum(A_zero_mean ** 2,repeat_index,dim=0), 1e-6))
    B_norm = torch.sqrt(torch.clamp_min(scatter_sum(B_zero_mean ** 2,repeat_index,dim=0), 1e-6))

    NCC = 1 - dot_product / (torch.clamp_min(A_norm * B_norm, 1e-6))
    times[1] += time.time() - cur_time
    cur_time = time.time()

    return NCC