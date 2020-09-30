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

# import net.net_utils as net_utils
from dataset.dataprocessing import DataProcessing

class SharedMLP(nn.Sequential):
    def __init__(self, args, bn, activation=True):
        super(SharedMLP, self).__init__()
        for i in range(len(args) - 1):
            self.add_module(
                'SharedMLP_conv2d_layer{}'.format(i),
                nn.Conv2d(args[i], args[i + 1], kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            if bn==True:
                self.add_module(
                    'SharedMLP_batchnorm_layer{}'.format(i),
                    nn.BatchNorm2d(args[i + 1], 1e-6, 0.99)
                )
            if activation==True:
                self.add_module(
                    'SharedMLP_activation_layer{}'.format(i),
                    nn.LeakyReLU(0.2, True)
                )


class AttentivePooling(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(AttentivePooling, self).__init__()
        self.AttentivePooling_fc = nn.Conv2d(channel_input, channel_input, (1, 1), bias=False)
        self.AttentivePooling_mlp = SharedMLP([channel_input, channel_output], bn=True)
    
    def forward(self, feature_set):
        feature_set = self.AttentivePooling_fc(feature_set) # feature_set:(batch_size, num_points, num_neighbour, channels)
        score = F.softmax(feature_set, dim=3) # sum(feature_set[batch_size][num_points][num_neighbour][i])=1
        weighted_score = feature_set.contiguous() * score
        weighted_score_sum = torch.sum(weighted_score, dim=3, keepdim=True)
        feature = self.AttentivePooling_mlp(weighted_score_sum)
        return feature


class BuildingBlock(nn.Module):
    def __init__(self, channel_out):
        super(BuildingBlock, self).__init__()
        self.BuildingBlock_mlp_1 = SharedMLP([10, channel_out // 2], bn=True)
        self.BuildingBlock_attentivepooling_1 = AttentivePooling(channel_out, channel_out // 2)
        self.BuildingBlock_mlp_2 = SharedMLP([channel_out // 2, channel_out // 2], bn=True)
        self.BuildingBlock_attentivepooling_2 = AttentivePooling(channel_out, channel_out)

    def forward(self, xyz, feature, neighbour_index):
        feature_xyz = self.relative_point_position_encoding(xyz, neighbour_index)  # feature_xyz:(batch_size, number_points, neighbour, 10)
        feature_xyz = feature_xyz.permute((0, 3, 1, 2))  #feature_xyz:(batch_size, 10, number_points, neighbour)
        feature_xyz = self.BuildingBlock_mlp_1(feature_xyz)
        feature_neighbour = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neighbour_index) # feature_neighbour:(batch_size,npoint,nsamples,channel)
        feature_neighbour = feature_neighbour.permute((0, 3, 1, 2)) #feature_neighbour:(batch_size, channel, number_points, neighbour)
        feature_concat = torch.cat([feature_neighbour, feature_xyz], dim=1)
        feature_pointcloud_aggregate = self.BuildingBlock_attentivepooling_1(feature_concat)

        feature_xyz = self.BuildingBlock_mlp_2(feature_xyz)
        feature_neighbour = self.gather_neighbour(feature_pointcloud_aggregate.squeeze(-1).permute((0, 2, 1)), neighbour_index)
        feature_neighbour = feature_neighbour.permute((0, 3, 1, 2))
        feature_concat = torch.cat([feature_neighbour, feature_xyz], dim=1)
        feature_pointcloud_aggregate = self.BuildingBlock_attentivepooling_2(feature_concat)
        return feature_pointcloud_aggregate

    def relative_point_position_encoding(self, xyz, neighbour_index):
        neighbour_xyz = self.gather_neighbour(xyz, neighbour_index)  # neighbour_xyz:(batch_size, number_points, neighbour, channel)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neighbour_index.shape[-1], 1) # xyz_tile:(batch_size, number_points, neighbour, channel)
        relative_xyz = xyz_tile - neighbour_xyz
        relative_distance = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdims=True)) # relative_distance:(batch_size, number_points, neighbour, 1)
        relative_feature = torch.cat([relative_distance, relative_xyz, xyz_tile, neighbour_xyz], dim=-1)  # relative_feature:(batch_size, number_points, neighbour, 10)
        return relative_feature

    def gather_neighbour(self, pointcloud, neighbour_index):
        batch_size = pointcloud.shape[0]
        number_points = pointcloud.shape[1]
        channel = pointcloud.shape[2]

        index_input = neighbour_index.reshape(batch_size, -1) # index_input:(batch_size, neighbour)
        index_input = index_input.unsqueeze(-1).repeat(1,1,pointcloud.shape[2]) # index_input:(batch_size, neighbour_index, channels)
        features = torch.gather(pointcloud, 1, index_input)
        features = features.reshape(batch_size, number_points, neighbour_index.shape[-1], channel)
        return features

class DilatedResBlock(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(DilatedResBlock, self).__init__()

        self.DilatedResBlock_mlp1 = SharedMLP([channel_input, channel_output // 2], bn=True)
        self.lfa = BuildingBlock(channel_output)
        self.DilatedResBlock_mlp2 = SharedMLP([channel_output, channel_output * 2], bn=True)
        self.DilatedResBlock_mlp3 = SharedMLP([channel_input, channel_output * 2], bn=True)
        
    def forward(self, feature, xyz, neighbour_index):
        feature_pointcloud = self.DilatedResBlock_mlp1(feature)
        feature_pointcloud = self.lfa(xyz, feature_pointcloud, neighbour_index)
        feature_pointcloud = self.DilatedResBlock_mlp2(feature_pointcloud)
        feature_shortcut = self.DilatedResBlock_mlp3(feature)
        return F.leaky_relu(feature_pointcloud + feature_shortcut, negative_slope=0.2)
        
class RandLANET(nn.Module):
    def __init__(self, dataset_name, config):
        super(RandLANET, self).__init__()
        self.config = config
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime(results_dir + '/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            os.makedirs(self.saving_path) if os.path.exists(self.saving_path) else None
        
        # weights
        self.class_weights = DataProcessing.get_class_weights(dataset_name)

        # net
        self.fc_0 = nn.Conv1d(3, 8, kernel_size=1, stride=1, bias=False)
        self.bn_0 = nn.BatchNorm1d(8, 1e-6, 0.99)
        self.activate_0 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        dimension_in = 8
        self.dilated_res_block = nn.ModuleList()
        for i in range(self.config.num_layers):
            dimension_out = self.config.dimension_out[i]
            self.dilated_res_block.append(DilatedResBlock(dimension_in, dimension_out))
            dimension_in = 2 * dimension_out
        
        dimension_out = dimension_in
        self.decoder_0 = SharedMLP([dimension_in, dimension_out], bn=True)
        
        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                dimension_in = dimension_out + 2* self.config.dimension_out[-j - 2]
                dimension_out = 2 * self.config.dimension_out[-j - 2]
            else:
                dimension_in = 4 * self.config.dimension_out[-4]
                dimension_out = 2 * self.config.dimension_out[-4]
            self.decoder_blocks.append(SharedMLP([dimension_in, dimension_out], bn=True))
            
        self.fc_1 = SharedMLP([dimension_out, 64], bn=True)
        self.fc_2 = SharedMLP([64, 32], bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_3 = SharedMLP([32, self.config.num_classes], bn=False, activation=False)

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
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
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
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def forward(self, end_points):
        features = end_points['features']
        features = self.fc_0(features)
        features = self.bn_0(features)
        features = self.activate_0(features)
        features = features.unsqueeze(dim=3)

        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block[i](features, end_points['xyz'][i], end_points['neighbour_index'][i])
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_index'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        features = self.decoder_0(f_encoder_list[-1])

        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interplation_index'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        
        features = self.fc_1(features)
        features = self.fc_2(features)
        features = self.dropout(features)
        features = self.fc_3(features)
        feature_out = features.squeeze(dim=3)

        end_points['logits'] = feature_out
        return end_points

    def get_loss(self, logits, labels, pre_class_weights):
        class_weights = torch.from_numpy(pre_class_weights).float().cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        output_loss = criterion(logits, labels)
        output_loss = output_loss.sum()
        return output_loss

    def compute_loss(self, end_points, cfg):
        logits = end_points['logits']
        labels = end_points['labels']

        logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = labels == 0
        for ign_label in cfg.ignored_label_index:
            ignored_bool = ignored_bool | (labels == ign_label)

        # Collect logits and labels that are not ignored
        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = labels[valid_idx]

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, cfg.num_classes).long().cuda()
        inserted_value = torch.zeros((1,)).long().cuda()
        for ign_label in cfg.ignored_label_index:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        loss = self.get_loss(valid_logits, valid_labels, cfg.class_weights)
        end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
        end_points['loss'] = loss
        return loss, end_points
    
    def compute_accuracy(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        logits = logits.max(dim=1)[1]
        acc = (logits == labels).sum().float() / float(labels.shape[0])
        end_points['accuracy'] = acc
        return acc, end_points

class IoUCalculator:
    def __init__(self, config):
        self.groundtruth_classes = [0 for _ in range(config.num_classes)]
        self.positive_classes = [0 for _ in range(config.num_classes)]
        self.true_positive_classes = [0 for _ in range(config.num_classes)]
        self.config = config

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        predict = logits.max(dim=1)[1]
        predict_valid = predict.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()
        
        val_total_correct = 0
        val_total_seen = 0
        correct = np.sum(predict_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        confidence_matrix = confusion_matrix(labels_valid, predict_valid, np.arange(0, self.config.num_classes, 1))
        self.groundtruth_classes += np.sum(confidence_matrix, axis=1)
        self.positive_classes += np.sum(confidence_matrix, axis=0)
        self.true_positive_classes += np.diagonal(confidence_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            if float(self.groudtruth_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.groudtruth_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list

