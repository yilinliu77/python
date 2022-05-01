import torch
import torchsort


def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


def loss_l2_error2(v_point_attribute, v_prediction, v_is_img_involved=False, v_use_scaled_loss=1.):
    predicted_recon_error = v_prediction[:, :, 0:1]
    predicted_gt_error = v_prediction[:, :, 1:2]

    smith_reconstructability = v_point_attribute[:, :, 0]

    gt_recon_error = v_point_attribute[:, :, 1:2]
    recon_mask = (gt_recon_error != -1).bool()
    gt_gt_error = v_point_attribute[:, :, 2:3]
    gt_mask = (gt_gt_error != -1).bool()

    scaled_gt_recon_error = torch.clamp(gt_recon_error, -v_use_scaled_loss, v_use_scaled_loss)
    scaled_gt_gt_error = torch.clamp(gt_gt_error, -v_use_scaled_loss, v_use_scaled_loss)

    recon_loss = torch.nn.functional.l1_loss(predicted_recon_error[recon_mask], scaled_gt_recon_error[recon_mask])
    gt_loss = torch.zeros_like(recon_loss)
    if v_is_img_involved:
        gt_loss = torch.nn.functional.l1_loss(predicted_gt_error[gt_mask], scaled_gt_gt_error[gt_mask])

    return recon_loss, gt_loss, gt_loss if v_is_img_involved else recon_loss

def loss_l2_error(v_point_attribute, v_prediction, v_is_img_involved=False, v_use_scaled_loss=1.):
    predicted_recon_error = v_prediction[:, :, 0:1]
    predicted_gt_error = v_prediction[:, :, 1:2]

    smith_reconstructability = v_point_attribute[:, :, 0]

    gt_recon_error = v_point_attribute[:, :, 1:2]
    recon_mask = (gt_recon_error != -1).bool()
    gt_gt_error = v_point_attribute[:, :, 2:3]
    gt_mask = (gt_gt_error != -1).bool()

    scaled_gt_recon_error = torch.clamp(gt_recon_error, -v_use_scaled_loss, v_use_scaled_loss)
    scaled_gt_gt_error = torch.clamp(gt_gt_error, -v_use_scaled_loss, v_use_scaled_loss)

    recon_loss = torch.nn.functional.l1_loss(predicted_recon_error[recon_mask], scaled_gt_recon_error[recon_mask])
    gt_loss = torch.zeros_like(recon_loss)
    if v_is_img_involved:
        scaled_gt_gt_error[torch.logical_not(gt_mask)] = v_use_scaled_loss
        gt_loss = torch.nn.functional.l1_loss(predicted_gt_error, scaled_gt_gt_error)

    return recon_loss, gt_loss, gt_loss if v_is_img_involved else recon_loss


def loss_spearman_error(v_point_attribute, v_prediction, v_is_img_involved=False, method="l2", normalized_factor=1.):
    predicted_recon_error = v_prediction[:, :, 0:1]
    predicted_gt_error = v_prediction[:, :, 1:2]

    smith_reconstructability = v_point_attribute[:, 0]

    gt_recon_error = v_point_attribute[:, :, 1:2]
    recon_mask = (gt_recon_error != -1).bool()
    gt_gt_error = v_point_attribute[:, :, 2:3]
    gt_mask = (gt_gt_error != -1).bool()

    scaled_gt_recon_error = torch.clamp(gt_recon_error, -1., 1.)
    scaled_gt_gt_error = torch.clamp(gt_gt_error, -1., 1.)

    recon_loss = spearmanr(
                predicted_recon_error[recon_mask].unsqueeze(0),
                gt_recon_error[recon_mask].unsqueeze(0),
                regularization=method, regularization_strength=normalized_factor
            )
    recon_loss = 1 - recon_loss  # We want to minimize the loss, which is maximizing the correlation factor
    gt_loss = torch.zeros_like(recon_loss)
    if v_is_img_involved:
        gt_loss = spearmanr(
            predicted_gt_error[gt_mask].unsqueeze(0),
            gt_gt_error[gt_mask].unsqueeze(0),
            regularization=method, regularization_strength=normalized_factor
        )
        gt_loss = 1 - gt_loss  # We want to minimize the loss, which is maximizing the correlation factor

    return recon_loss, gt_loss, gt_loss if v_is_img_involved else recon_loss

