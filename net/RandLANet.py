#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
import sklearn
import net.net_utils as net_utils

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

from dataset.dataprocessing import DataProcessing

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

class AttentivePooling(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(AttentivePooling, self).__init__()
        self.AttentivePooling_fc = nn.Conv2d(channel_input, channel_input, (1, 1), bias=False)
        self.AttentivePooling_mlp = net_utils.Conv2d(channel_input, channel_output, kernel_size=(1, 1), bn=True)
    
    def forward(self, feature_set):
        feature_set = self.AttentivePooling_fc(feature_set) # feature_set:(batch_size, num_points, num_neighbour, channels)
        score = F.softmax(feature_set, dim=3) # sum(feature_set[batch_size][num_points][num_neighbour][i])=1
        weighted_score = feature_set * score
        weighted_score_sum = torch.sum(weighted_score, dim=3, keepdim=True)
        feature = self.AttentivePooling_mlp(weighted_score_sum)
        return feature

class BuildingBlock(nn.Module):
    def __init__(self, channel_out):
        super(BuildingBlock, self).__init__()
        self.BuildingBlock_mlp_1 = net_utils.Conv2d(10, channel_out // 2, kernel_size=(1, 1), bn=True)
        self.BuildingBlock_attentivepooling_1 = AttentivePooling(channel_out, channel_out // 2)
        self.BuildingBlock_mlp_2 = net_utils.Conv2d(channel_out // 2, channel_out // 2, kernel_size=(1, 1), bn=True)
        self.BuildingBlock_attentivepooling_2 = AttentivePooling(channel_out, channel_out)

    def forward(self, xyz, feature, neighbour_index):
        feature_xyz = self.relative_point_position_encoding(xyz, neighbour_index)  # feature_xyz:(batch_size, number_points, neighbour, 10)
        feature_xyz = feature_xyz.permute((0, 3, 1, 2))  #feature_xyz:(batch_size, 10, number_points, neighbour)
        feature_xyz = self.BuildingBlock_mlp_1(feature_xyz)
        feature_neighbour = self.gather_neigbour(feature.squeeze(-1).permute((0, 2, 1), neighbour_index)) # feature_neighbour:(batch_size,npoint,nsamples,channel)
        feature_neighbour = feature_neighbour.permute((0, 3, 1, 2)) #feature_neighbour:(batch_size, channel, number_points, neighbour)
        feature_concat = torch.cat([feature_neighbour, feature_xyz], dim=1)
        feature_pointcloud_aggregate = self.BuildingBlock_attentivepooling_1(feature_concat)

        feature_xyz = self.BuildingBlock_mlp_2(feature_xyz)
        feature_neighbour = self.gather_neigbour(feature_pointcloud_aggregate.squeeze(-1).permute((0, 2, 1)), neighbour_index)
        feature_neighbour = feature_neighbour.permute((0, 3, 1, 2))
        feature_concat = torch.cat([feature_neighbour, feature_xyz], dim=1)
        feature_pointcloud_aggregate = self.BuildingBlock_attentivepooling_2(feature_concat)
        return feature_pointcloud_aggregate

    def relative_point_position_encoding(self, xyz, neighbour_index):
        neighbour_xyz = self.gather_neigbour(xyz, neighbour_index)  # neighbour_xyz:(batch_size, number_points, neighbour, channel)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neighbour_index.shape[-1], 1) # xyz_tile:(batch_size, number_points, neighbour, channel)
        relative_xyz = xyz_tile - neighbour_xyz
        relative_distance = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdims=True)) # relative_distance:(batch_size, number_points, neighbour, 1)
        relative_feature = torch.cat([relative_distance, relative_xyz, xyz_tile, neighbour_xyz], dim=-1)  # relative_feature:(batch_size, number_points, neighbour, 10)
        return relative_feature

    def gather_neigbour(self, pointcloud, neighbour_index):
        batch_size = pointcloud.shape[0]
        number_points = pointcloud.shape[1]
        channel = pointcloud.shape[2]

        index_input = neighbour_index.reshape(batch_size, -1) # index_input:(batch_size, neighbours)
        index_input = index_input.unsqueeze(-1).repeat(1,1,pointcloud.shape[2]) # index_input:(batch_size, neighbour_index, channels)
        features = torch.gather(pointcloud, 1, index_input)
        features = features.reshape(batch_size, number_points, neighbour_index.shape[-1], channel)
        return features

class DilatedResBlock(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(DilatedResBlock, self).__init__()

        self.DilatedResBlock_mlp1 = net_utils.Conv2d(channel_input, channel_output // 2, kernel_size=(1, 1), bn=True)
        self.lfa = BuildingBlock(channel_output)
        self.DilatedResBlock_mlp2 = net_utils.Conv2d(channel_output, channel_output * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.DilatedResBlock_mlp3 = net_utils.Conv2d(channel_input, channel_output * 2, kernel_size=(1, 1), bn=True, acivation=None)
        
    def forward(self, feature, xyz, neighbour_index):
        feature_pointcloud = self.DilatedResBlock_mlp1(feature)
        feature_pointcloud = self.lfa(xyz, feature_pointcloud, neighbour_index)
        feature_pointcloud = self.DilatedResBlock_mlp2(feature_pointcloud)
        feature_shortcut = self.DilatedResBlock_mlp3(feature)
        return F.leaky_relu(feature_pointcloud + feature_shortcut, negative_slope=0.2)
        
class RandLANET(nn.Module):
    def __init__(self, dataset, config):
        super(RandLANET, self).__init__()
        self.config = config
        if self.config.save:
            if self.config.save_path is None:
                self.save_path = time.strftime(results_dir + '/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.save_path = self.config.save_path
            os.makedirs(self.save_path) if os.path.exists(self.save_path) else None
        
        # input
        flat_inputs = dataset.flat_inputs
        self.input = dict()
        num_layers = self.config.num_layers
        self.inputs['xyz'] = flat_inputs[:num_layers]
        self.inputs['neighbour_index'] = flat_inputs[num_layers: 2 * num_layers]
        self.inputs['sub_index'] = flat_inputs[2 * num_layers:3 * num_layers]
        self.inputs['interpolation_index'] = flat_inputs[3 * num_layers:4 * num_layers]
        self.inputs['features'] = flat_inputs[4 * num_layers]
        self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
        self.inputs['input_indexs'] = flat_inputs[4 * num_layers + 2]
        self.inputs['cloud_indexs'] = flat_inputs[4 * num_layers + 3]
        
        self.labels = self.inputs['labels']
        self.class_weights = DataProcessing.get_class_weights(dataset.name)
        self.log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')
        
        # net
        self.fc_0 = net_utils.Conv1d(3, 8, kernel_size=1, bn=True)
        dimension_in = 8
        self.dilated_res_block = nn.ModuleList()
        for i in range(self.config.num_layers):
            dimension_out = self.config.dimension_out[i]
            self.dilated_res_block.append(DilatedResBlock(dimension_in, dimension_out))
            dimension_in = 2 * dimension_out
        
        dimension_out = dimension_in
        self.decoder_0 = net_utils.Conv2d(dimension_in, dimension_out, kernel_size=(1, 1), bn=True)
        
        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                dimension_in = dimension_out + self.config.dimension_out[-j - 2]
                dimension_out = 2 * self.config.dimension_out[-j - 2]
            else:
                dimension_in = 4 * self.config.dimension_out[-4]
                dimension_out = 2 * self.config.dimension_out[-4]
            self.decoder_blocks.append(net_utils.Conv2d(dimension_in, dimension_out, kernel_size=(1, 1), bn=True))
            
        self.fc_1 = net_utils.Conv2d(dimension_out, 64, kernel_size=(1, 1), bn=True)
        self.fc_2 = net_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_3 = net_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def random_sample(self, feature, pool_index):
        feature = feature.unsqueeze(3)
        num_neighbour = pool_index.shape[-1]
        dimension = feature.shape[1]
        batch_size = pool_index.shape[0]

        pool_index = pool_index.reshape(batch_size, -1)
        pool_feature = torch.gather(feature, 2, pool_index.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_feature = pool_feature.reshape(batch_size, dimension, -1, num_neighbour)
        pool_feature = pool_feature.max(dim=3, keepdims=True)[0]
        return pool_feature

    def nearest_interpolation(self, feature, interpolation_index):
        feature = feature.unsequeeze(3)
        batch_size = interpolation_index.shape[0]
        upsample_num_points = interpolation_index.shape[1]

        interpolation_index = interpolation_index.reshape(batch_size, upsample_num_points)

        interpolation_features = torch.gather(feature, 2, interpolation_index.unsequeeze(1).repeat(1, feature.shape[1], 1))
        interpolation_features = interpolation_features.unsequeeze(3)
        return interpolation_features

    def forward(self, input):
        features = input['features']
        features = self.fc_0(features)
        features = features.unsqueeze(3)

        features_encoder_list = []
        for i in range(self.config.num_layers):
            features_encoder_i = self.dilated_res_block[i](features, input['xyz'][i], input['neighbour_index'][i])
            features_sampled_i = self.random_sample(features_encoder_i, input['sub_index'][i])
            features = features_sampled_i
            
            if i == 0:
                features_encoder_list.append(features_encoder_i)
            features_encoder_list.append(features_sampled_i)

        features = self.decoder_0(features_encoder_list[-1])

        features_decoder_list = []
        for j in range(self.config.num_layers):
            features_interpolation_i = self.nearest_interpolation(features, input['interpolation_index'][-j - 1])
            features_decoder_i = self.decoder_blocks[j](torch.cat([features_encoder_list[-j - 2], features_interpolation_i], dim=1))
            features = features_decoder_i
            features_decoder_list.append(features_decoder_i)
        
        fetures = self.fc_1(features)
        features = self.fc_2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        feature_out = features.sequeeze(3)

        input['logits'] = feature_out
        return input

    def get_loss(self, logits, labels, pre_class_weights):
        class_weights = torch.from_numpy(pre_class_weights).float().cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        output_loss = criterion(logits, labels)
        output_loss = output_loss.sum()
        return output_loss

    def compute_loss(self, input, config):
        logits = input['logits'].transpose(1,2).reshape(-1, config.num_classes)
        labels = input['labels'].reshape(-1)

        ignored_bool = labels = 0
        for ignored_label in config.ignored_label_index:
            ignored_bool = ignored_bool | (labels == ignored_label)
        
        valid_index = ignored_bool == 0
        valid_logits = logits[valid_index, :]
        valid_labels_init = labels[valid_index]

        

