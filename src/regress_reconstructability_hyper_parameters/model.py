import torch
import torchvision
from scipy.stats import stats
from torch import nn
import numpy as np
import torch.nn.functional as F

from fast_soft_sort.pytorch_ops import soft_rank
from torchvision.models import resnet18

import sys

from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, \
    PointNetFeaturePropagation


def spearmanr(pred, target, **kw):
    pred = soft_rank(pred.cpu(), **kw).cuda()
    target = soft_rank(target.cpu(), **kw).cuda()
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()



def normal_distribution(v_x, v_sigma, v_exponential):
    return torch.exp(-torch.pow(v_x, v_exponential) / (2 * v_sigma * v_sigma))


class Regress_hyper_parameters_Model(nn.Module):
    def __init__(self, hparams):
        super(Regress_hyper_parameters_Model, self).__init__()
        self.hydra_conf = hparams

        self.alpha_weights_sigma1 = nn.Parameter(torch.tensor(5. / 90, dtype=torch.float))
        self.alpha_weights_sigma2 = nn.Parameter(torch.tensor(30. / 90, dtype=torch.float))
        self.alpha_weights_exponential1 = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)
        self.alpha_weights_exponential2 = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.gsd_weights_sigma = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
        self.gsd_weights_exponential = nn.Parameter(torch.tensor(4., dtype=torch.float), requires_grad=False)

        self.scale_weights_sigma = nn.Parameter(torch.tensor(.25, dtype=torch.float))
        self.scale_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.angle_to_normal_weights_sigma = nn.Parameter(torch.tensor(30. / 90, dtype=torch.float))
        self.angle_to_normal_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float),
                                                                requires_grad=False)

        self.distortion_weights_sigma = nn.Parameter(torch.tensor(1., dtype=torch.float))
        self.distortion_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.model_weights_sigma = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.model_weights_exponential = nn.Parameter(torch.tensor(2, dtype=torch.float), requires_grad=False)

    def forward(self, v_data):
        # alpha_in_degree, d1/d_max, d2/d_max, 1-min(d1,d2)/max(d1,d2), normal_angle1, normal_angle2, central_angle1, central_angle2
        views = v_data["views"]
        view_pairs = v_data["view_pairs"]
        point_attribute = v_data["point_attribute"]

        mask = v_data["views"][:, :, 0] == 1

        step_alpha_sigma = torch.zeros_like(view_pairs[:, :, 1])
        step_alpha_sigma[view_pairs[:, :, 1] < 20. / 90] = self.alpha_weights_sigma1
        step_alpha_sigma[view_pairs[:, :, 1] > 20. / 90] = self.alpha_weights_sigma2
        alpha_weights_exponential = torch.zeros_like(view_pairs[:, :, 1])
        alpha_weights_exponential[view_pairs[:, :, 1] < 20. / 90] = self.alpha_weights_exponential1
        alpha_weights_exponential[view_pairs[:, :, 1] > 20. / 90] = self.alpha_weights_exponential2

        base_score = normal_distribution(view_pairs[:, :, 1],
                                         step_alpha_sigma, alpha_weights_exponential)

        gsd_score = normal_distribution(views[:, :, 4], self.gsd_weights_sigma, self.gsd_weights_exponential)
        scale_score = normal_distribution(view_pairs[:, :, 2], self.scale_weights_sigma, self.scale_weights_exponential)
        normal_score = normal_distribution(views[:, :, 5],
                                           self.angle_to_normal_weights_sigma,
                                           self.angle_to_normal_weights_exponential)
        distortion_score = normal_distribution(views[:, :, 6],
                                               self.distortion_weights_sigma, self.distortion_weights_exponential)

        # Filter out the invalid view
        view_pairs_mask = view_pairs[:, :, 0] != 0
        base_score = base_score * view_pairs_mask
        gsd_score = gsd_score * mask
        scale_score = scale_score * view_pairs_mask
        normal_score = normal_score * mask
        distortion_score = distortion_score * mask

        # Sum up
        flatten_mask = torch.triu_indices(gsd_score.shape[1], gsd_score.shape[1], offset=1)

        gsd_score = torch.einsum('bi,bj->bij', (gsd_score, gsd_score))
        gsd_score = torch.triu(gsd_score, diagonal=1)
        gsd_score = gsd_score[:, flatten_mask[0], flatten_mask[1]]

        normal_score = torch.einsum('bi,bj->bij', (normal_score, normal_score))
        normal_score = torch.triu(normal_score, diagonal=1)
        normal_score = normal_score[:, flatten_mask[0], flatten_mask[1]]

        distortion_score = torch.einsum('bi,bj->bij', (distortion_score, distortion_score))
        distortion_score = torch.triu(distortion_score, diagonal=1)
        distortion_score = distortion_score[:, flatten_mask[0], flatten_mask[1]]

        # final_score = base_score
        final_score = base_score * gsd_score * scale_score * normal_score * distortion_score
        final_score = torch.sum(final_score, dim=1)
        # final_score_exp = normal_distribution(final_score,self.model_weights_sigma,self.model_weights_exponential)
        return final_score

    def loss(self, v_point_attribute, v_prediction):
        # v_error = torch.clamp_max(v_error,1.)
        gt_reconstructability = v_point_attribute[:,0]
        gt_error = v_point_attribute[:,1]
        valid_point = gt_error < 9999999
        num_valid_point = valid_point.sum()

        valid_reconstructability_exp = v_prediction[valid_point]
        valid_error_exp = gt_error[valid_point]

        soft_spearman = spearmanr(valid_reconstructability_exp.unsqueeze(0), valid_error_exp.unsqueeze(0),
                                  regularization_strength=1e-2,
                                  regularization="kl")
        gt_spearman = stats.spearmanr(valid_reconstructability_exp.detach().cpu().numpy(),
                                      valid_error_exp.detach().cpu().numpy())[0]

        return soft_spearman, gt_spearman, num_valid_point



class Brute_force_nn(nn.Module):
    def __init__(self, hparams):
        super(Brute_force_nn, self).__init__()
        self.hydra_conf = hparams

        self.reconstructability_predictor = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.reconstructability_to_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m,(nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        batch_size = v_data["views"].shape[0]
        if len(v_data["views"].shape) == 4:
            view_attribute = v_data["views"].view(-1, v_data["views"].shape[2], v_data["views"].shape[3])
        else:
            view_attribute = v_data["views"]
        view_attribute[:,:,1:4] = view_attribute[:,:,1:4] / (torch.norm(view_attribute[:,:,1:4],dim=2).unsqueeze(-1)+1e-6)
        predict_reconstructability_per_view = self.reconstructability_predictor(
            view_attribute[:,:,1:7]) # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view) # Filter out the unusable view
        valid_mask[view_attribute[:,:,0].bool()] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view*valid_mask
        predict_reconstructability = torch.sum(predict_reconstructability_per_view,dim=1) # Sum up all the view contribution of one point
        predict_reconstructability = predict_reconstructability / torch.sum(view_attribute[:,:,0].bool(),dim=1).unsqueeze(-1) # Normalize the features
        predict_error = self.reconstructability_to_error(predict_reconstructability) # Map it to point error
        if predict_error.shape[0]!=batch_size:
            predict_error=predict_error.reshape(batch_size,-1,predict_error.shape[1])
        return predict_error

    def loss(self, v_point_attribute, v_prediction):
        gt_reconstructability = v_point_attribute[:,0]
        gt_error = v_point_attribute[:,1:2]

        loss = torch.nn.functional.mse_loss(v_prediction,gt_error)
        gt_spearman = stats.spearmanr(v_prediction.detach().cpu().numpy(),
                                      gt_error.detach().cpu().numpy())[0]

        return loss, gt_spearman, 0


class Correlation_nn(nn.Module):
    def __init__(self, hparams):
        super(Correlation_nn, self).__init__()
        self.hydra_conf = hparams
        self.view_feature_extractor=nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LeakyReLU(),
                    # nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                    # nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                )
        self.view_feature_fusioner1=nn.MultiheadAttention(embed_dim=256,num_heads=1)
        self.view_feature_fusioner_linear1=nn.Linear(256, 256)
        self.view_feature_fusioner_relu1=nn.LeakyReLU()
        self.view_feature_fusioner2=nn.MultiheadAttention(embed_dim=256,num_heads=1)
        self.view_feature_fusioner_linear2=nn.Linear(256, 256)
        self.view_feature_fusioner_relu2=nn.LeakyReLU()

        self.reconstructability_to_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        for m in self.parameters():
            if isinstance(m,(nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        batch_size = v_data["views"].shape[0]
        if len(v_data["views"].shape) == 4:
            view_attribute = v_data["views"].view(-1,v_data["views"].shape[2],v_data["views"].shape[3])
        else:
            view_attribute = v_data["views"]
        view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
                    torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:7])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].bool()] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        predict_reconstructability=[]
        for i in range(predict_reconstructability_per_view.shape[0]//1024+1):
            if i*1024 == predict_reconstructability_per_view.shape[0]:
                break
            feature_item = predict_reconstructability_per_view[i*1024:min(i*1024+1024,predict_reconstructability_per_view.shape[0])]
            feature_item=torch.transpose(feature_item,0,1)
            #
            attention_result = self.view_feature_fusioner1(feature_item,feature_item,feature_item)
            attention_result = self.view_feature_fusioner_linear1(torch.transpose(attention_result[0],0,1))
            attention_result = self.view_feature_fusioner_relu1(attention_result)
            feature_item=torch.transpose(attention_result,0,1)
            attention_result = self.view_feature_fusioner2(feature_item,feature_item,feature_item)
            attention_result = self.view_feature_fusioner_linear2(torch.transpose(attention_result[0],0,1))
            attention_result = self.view_feature_fusioner_relu2(attention_result)

            predict_reconstructability.append(attention_result)
        predict_reconstructability=torch.cat(predict_reconstructability,dim=0)
        predict_reconstructability = torch.sum(predict_reconstructability,
                                               dim=1)  # Sum up all the view contribution of one point
        predict_reconstructability = predict_reconstructability / torch.sum(view_attribute[:, :, 0].bool(),
                                                                            dim=1).unsqueeze(
            -1)  # Normalize the features

        predict_error = self.reconstructability_to_error(predict_reconstructability)  # Map it to point error
        if predict_error.shape[0]!=batch_size:
            predict_error=predict_error.reshape(batch_size,-1,predict_error.shape[1])
        return predict_error

    def loss(self, v_point_attribute, v_prediction):
        gt_reconstructability = v_point_attribute[:,0]
        gt_error = v_point_attribute[:,1:2]

        loss = torch.nn.functional.mse_loss(v_prediction,gt_error)
        gt_spearman = stats.spearmanr(v_prediction.detach().cpu().numpy(),
                                      gt_error.detach().cpu().numpy())[0]

        return loss, gt_spearman, 0



class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 35 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return x, l4_points


class Uncertainty_Modeling(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        sys.path.insert(0,r"thirdparty\Pointnet_Pointnet2_pytorch")
        from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import get_model
        self.point_feature_extractor = PointNet(13)
        sys.path.pop()

        # self.img_feature_extractor = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

        # self.geometrical_feature_extractor = Correlation_nn(hparams)
        self.geometrical_feature_extractor = Brute_force_nn(hparams)

        self.view_feature_fusioner1 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear1 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu1 = nn.LeakyReLU()
        self.view_feature_fusioner2 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear2 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu2 = nn.LeakyReLU()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        points = v_data["points"]
        if self.is_involve_img:
            img_features=[]
            for id_batch,item_batch in enumerate(v_data["point_features"]):
                pixel_position_features = item_batch

                # Fuse the image features
                pixel_position_features=pixel_position_features.transpose(0, 1)
                pixel_position_features = self.view_feature_fusioner1(pixel_position_features,
                                                                      pixel_position_features,
                                                                      pixel_position_features,
                                                                      # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                      key_padding_mask = v_data["point_features_mask"][id_batch])[0]
                pixel_position_features=self.view_feature_fusioner_linear1(pixel_position_features.transpose(0,1))
                pixel_position_features=self.view_feature_fusioner_relu1(pixel_position_features)
                pixel_position_features=pixel_position_features.transpose(0, 1)
                pixel_position_features = self.view_feature_fusioner2(pixel_position_features,
                                                                      pixel_position_features,
                                                                      pixel_position_features,
                                                                      # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                      key_padding_mask = v_data["point_features_mask"][id_batch])[0]
                pixel_position_features=self.view_feature_fusioner_linear2(pixel_position_features.transpose(0, 1))
                pixel_position_features=self.view_feature_fusioner_relu2(pixel_position_features)

                img_features.append(torch.mean(pixel_position_features,dim=1))

            img_features = torch.stack(img_features,dim=0)
            point_features = torch.cat([points[:,:,:3],img_features],dim=2)

            uncertainty = self.point_feature_extractor(point_features.transpose(1,2))[0].transpose(1,2)

        else:
            uncertainty = 1

        geometrical_feature = self.geometrical_feature_extractor(v_data)

        return geometrical_feature * uncertainty

    def loss(self, v_point_attribute, v_prediction):
        gt_reconstructability = v_point_attribute[:, 0]
        gt_max_error = v_point_attribute[:,:, 1:2]
        gt_avg_error = v_point_attribute[:,:, 2:3]

        loss = torch.nn.functional.mse_loss(v_prediction, gt_avg_error)
        # gt_spearmans=[]
        # for id_item in range(v_point_attribute.shape[0]):
        #     gt_spearman = stats.spearmanr(v_prediction[id_item,:,0].detach().cpu().numpy(),
        #                                   gt_avg_error[id_item,:,0].detach().cpu().numpy())[0]
        #     gt_spearmans.append(gt_spearman)

        # return loss, np.mean(gt_spearmans), 0
        return loss, 0, 0