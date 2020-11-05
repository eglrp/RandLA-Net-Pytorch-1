#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
from sklearn.metrics import confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
utils_dir = os.path.join(root_dir, 'utils')
dataset_dir = os.path.join(root_dir, 'dataset')
results_dir = os.path.join(root_dir, 'results')
sys.path.append(base_dir)
sys.path.append(root_dir)
sys.path.append(utils_dir)
sys.path.append(dataset_dir)
sys.path.append(results_dir)

import net.net_utils as net_utils
from dataset.dataprocessing import DataProcessing


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = net_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = net_utils.Conv2d(10,
                                     d_out // 2,
                                     kernel_size=(1, 1),
                                     bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = net_utils.Conv2d(d_out // 2,
                                     d_out // 2,
                                     kernel_size=(1, 1),
                                     bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature,
                neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(
            xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(
            feature.squeeze(-1).permute((0, 2, 1)),
            neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute(
            (0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(
            f_pc_agg.squeeze(-1).permute((0, 2, 1)),
            neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute(
            (0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(
            xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1],
                                           1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1,
                      keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat(
            [relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
            dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    def gather_neighbour(self, pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(
            pc, 1,
            index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points,
                                    neighbor_idx.shape[-1],
                                    d)  # batch*npoint*nsamples*channel
        return features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = net_utils.Conv2d(d_in,
                                     d_out // 2,
                                     kernel_size=(1, 1),
                                     bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = net_utils.Conv2d(d_out,
                                     d_out * 2,
                                     kernel_size=(1, 1),
                                     bn=True,
                                     activation=None)
        self.shortcut = net_utils.Conv2d(d_in,
                                         d_out * 2,
                                         kernel_size=(1, 1),
                                         bn=True,
                                         activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class RandLANET(nn.Module):
    def __init__(self, config):
        super(RandLANET, self).__init__()
        self.config = config
        # self.class_weights = DataProcessing.get_class_weights(dataset_name)

        # net
        self.fc0 = net_utils.Conv1d(config.channels, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]  # 16, 64, 128, 256
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            # semantickitti: (8,16), (32,64), (128,128), (256,256)
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = net_utils.Conv2d(d_in,
                                          d_out,
                                          kernel_size=(1, 1),
                                          bn=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, config.num_classes)

        # self.decoder_blocks = nn.ModuleList()
        # for j in range(self.config.num_layers):
        #     if j < 3:
        #         d_in = self.config.d_out[-j -
        #                                  1] * 2 + 2 * self.config.d_out[-j - 2]
        #         d_out = 2 * self.config.d_out[-j - 2]
        #     else:
        #         d_in = self.config.d_out[-j - 1] * 2 + 2 * self.config.d_out[0]
        #         d_out = 2 * self.config.d_out[0]
        #     self.decoder_blocks.append(
        #         net_utils.Conv2d_Transpose(d_in,
        #                                    d_out,
        #                                    kernel_size=(1, 1),
        #                                    bn=True))

        # self.fc1 = net_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        # self.fc2 = net_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        # self.dropout = nn.Dropout(0.5)
        # self.fc3 = net_utils.Conv2d(32,
        #                             self.config.num_classes,
        #                             kernel_size=(1, 1),
        #                             bn=False,
        #                             activation=None)

    def random_sample(self, feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2,
            pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(
            dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    def nearest_interpolation(self, feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2,
            interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(
            3)  # batch*channel*npoints*1
        return interpolated_features

    def features(self, data):
        features = data  # (batch, 3, N)
        features = self.fc0(features)  # (batch, 8, N)
        features = features.unsqueeze(dim=3)  # (batch, 8, N, 1)
        return features

    def encoder_decoder(self, features, xyz, neigh_idx, sub_idx, interp_idx):
        # features.shape # (batch, 8, N, 1)
        # xyz.shape # (batch,N,3)
        # neigh_idx # (batch,N,16)
        # sub_idx # (batch,N/4,16)
        # interp_idx # (batch,N,1)
        # ###########################Encoder############################
        # ok
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, xyz[i],
                                                     neigh_idx[i])
            # f_encoder_i.shape: (batch, 2*d_out[0], N, 1), (batch, 2*d_out[1], N/4, 1), ...
            # in tensorflow: (?,?,1,2*d_out[0]), (?,?,1,2*d_out[1]), ...

            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            # f_sampled_i.shape: (batch, 2*d_out[0], N/4, 1), (batch, 2*d_out[1], N/16, 1), ...
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            # print('0',f_encoder_i.shape)
        ###########################Encoder############################
        features = self.decoder_0(f_encoder_list[-1])
        features = torch.max(features, 2)[0]
        # print('shape', features.shape)
        x = features.squeeze()
        # shape: (batch, 2*d_out[-1], N/128, 1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

        # ###########################Decoder############################
        # f_decoder_list = []
        # for j in range(self.config.num_layers):
        #     f_interp_i = self.nearest_interpolation(features,
        #                                             interp_idx[-j - 1])
        #     # print('1', f_encoder_list[-j - 2].shape)
        #     # print('2', f_interp_i.shape)
        #     f_decoder_i = self.decoder_blocks[j](torch.cat(
        #         [f_encoder_list[-j - 2], f_interp_i], dim=1))
        #     # print('3', f_decoder_i.shape)
        #     features = f_decoder_i
        #     f_decoder_list.append(f_decoder_i)
        # # ###########################Decoder############################

        # features = self.fc1(features)
        # features = self.fc2(features)
        # features = self.dropout(features)
        # features = self.fc3(features)
        # f_out = features.squeeze(3)
        # return f_out

    def forward(self, xyz, neigh_idx, sub_idx, interp_idx, features, labels,
                input_inds, cloud_inds):

        feature = self.features(features)  # (batch, 8, N, 1)
        f_out = self.encoder_decoder(feature, xyz, neigh_idx, sub_idx,
                                     interp_idx)

        return f_out


def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss


def compute_loss(logits, labels, cfg):
    # logits = end_points['logits']
    # labels = end_points['labels']

    # logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    # labels = labels.reshape(-1)

    # # Boolean mask of points that should be ignored
    # ignored_bool = labels == 0
    # for ign_label in cfg.ignored_label_inds:
    #     ignored_bool = ignored_bool | (labels == ign_label)

    # # Collect logits and labels that are not ignored
    # valid_idx = ignored_bool == 0
    # valid_logits = logits[valid_idx, :]
    # valid_labels_init = labels[valid_idx]

    # # Reduce label values in the range of logit shape
    # reducing_list = torch.arange(0, cfg.num_classes).long().cuda()
    # inserted_value = torch.zeros((1, )).long().cuda()
    # for ign_label in cfg.ignored_label_inds:
    #     reducing_list = torch.cat([
    #         reducing_list[:ign_label], inserted_value,
    #         reducing_list[ign_label:]
    #     ], 0)
    # valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    # loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    # # end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    # # end_points['loss'] = loss
    # return loss, valid_logits, valid_labels
    total_loss = F.nll_loss(logits, labels)
    return total_loss


def compute_acc(logits, labels):
    # logits = end_points['valid_logits']
    # labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    # end_points['acc'] = acc
    return acc


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, logits, labels):
        # logits = end_points['valid_logits']
        # labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid,
                                       np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] -
                     self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(
                    self.gt_classes[n] + self.positive_classes[n] -
                    self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list