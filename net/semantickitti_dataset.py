#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from dataset.dataprocessing import DataProcessing
from config.config_semantickitti import Config_SemanticKITTI

class SemanticKITTI(torch_data.Dataset):
    def __init__(self, mode, test_id=None):
        self.name = 'SemanticKITTI'
        self.dataset_path = os.path.join(root_dir, 'data/semantickitti/dataset/sequences_0.06')
        self.label_to_names = {0: 'unlabeled',
                                1: 'car',
                                2: 'bicycle',
                                3: 'motorcycle',
                                4: 'truck',
                                5: 'other-vehicle',
                                6: 'person',
                                7: 'bicyclist',
                                8: 'motorcyclist',
                                9: 'road',
                                10: 'parking',
                                11: 'sidewalk',
                                12: 'other-ground',
                                13: 'building',
                                14: 'fence',
                                15: 'vegetation',
                                16: 'trunk',
                                17: 'terrain',
                                18: 'pole',
                                19: 'traffic-sign'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_index = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.seq_list = np.sort(os.listdir(self.dataset_path))

        if mode == 'test':
            self.test_scan_number = str(test_id)
        
        self.mode = mode
        train_list, val_list, test_list = DataProcessing.get_file_list(self.dataset_path, str(test_id))
        if mode == 'training':
            self.data_list = train_list
        elif mode == 'validation':
            self.data_list = val_list
        elif mode == 'test':
            self.data_list = test_list

        self.data_list = DataProcessing.shuffle_list(self.data_list)

        self.possibility = []
        self.min_possibility = []

        if mode == 'test':
            self.path_list = self.data_list
            for test_file_name in self.path_list:
                points = np.load(test_file_name)
                self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                self.min_possibility += [float(np.min(self.possibility[-1]))]

        Config_SemanticKITTI.ignored_label_inds = [self.label_to_index[ign_label] for ign_label in self.ignored_labels]
        Config_SemanticKITTI.class_weights = DataProcessing.get_class_weights('SemanticKITTI')
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        selected_pointcloud, selected_labels, selected_index, cloud_index = self.spatially_regular_gen(item)
        return selected_pointcloud, selected_labels, selected_index, cloud_index

    def spatially_regular_gen(self, item):
        # Generator loop
        if self.mode != 'test':
            cloud_index = item
            pointcloud_path = self.data_list[cloud_index]
            pointcloud, tree, labels = self.get_data(pointcloud_path)
            pick_index = np.random.choice(len(pointcloud), 1)
            selected_pointcloud, selected_labels, selected_index = self.crop_pointcloud(pointcloud, labels, tree, pick_index)
        
        else:
            cloud_index = int(np.argmin(self.min_possibility))
            pick_index = np.argmin(self.possibility[cloud_index])
            pointcloud_path = self.path_list[cloud_index]
            pointcloud, tree, labels = self.get_data(pointcloud_path)
            selected_pointcloud, selected_labels, selected_index = self.crop_pointcloud(pointcloud, labels, tree, pick_index)

            # update the possibility of the selected pc
            dists = np.sum(np.square((selected_pointcloud - pointcloud[pick_index]).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[cloud_index][selected_index] += delta
            self.min_possibility[cloud_index] = np.min(self.possibility[cloud_index])
        
        return selected_pointcloud.astype(np.float32), selected_labels.astype(np.int32), selected_index.astype(np.int32), np.array([cloud_index], dtype=np.int32)

    def get_data(self, file_path):
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = os.path.join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)

        if int(seq_id) >= 11:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path = os.path.join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
        
        return points, search_tree, labels

    def crop_pointcloud(self, points, labels, search_tree, pick_index):
        center_point = points[pick_index, :].reshape(1, -1)
        select_index = DataProcessing.shuffle_index(search_tree.query(center_point, k=Config_SemanticKITTI.num_points)[1][0])
        select_points = points[select_index]
        select_labels = labels[select_index]

        return select_points, select_labels, select_index

    def tf_map(self, batch_pointcloud, batch_label, batch_pointcloud_index, batch_cloud_index):
        features = batch_pointcloud
        input_points = []
        input_neighbours = []
        input_pools = []
        input_up_samples = []

        for i in range(Config_SemanticKITTI.number_layers):
            neighbour_index = DataProcessing.knn_search(batch_pointcloud, batch_pointcloud, Config_SemanticKITTI.knn)
            sub_points = batch_pointcloud[:, : batch_pointcloud.shape[1] // Config_SemanticKITTI.sub_sampling_ratio[i], :]
            pool_i = neighbour_index[:, : batch_pointcloud.shape[1] // Config_SemanticKITTI.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_pointcloud, 1)
            input_points.append(batch_pointcloud)
            input_neighbours.append(neighbour_index)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pointcloud = sub_points
    
        input_list = input_points + input_neighbours + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pointcloud_index, batch_cloud_index]
    
        return input_list

    def collate_fn(self, batch):
        selected_pointcloud, selected_labels, selected_index, cloud_index = [], [], [], []
        for i in range(len(batch)):
            print(i)
            selected_pointcloud.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_index.append(batch[i][2])
            cloud_index.append(batch[i][3])
        
        selected_pointcloud = np.stack(selected_pointcloud)
        selected_labels = np.stack(selected_labels)
        selected_index = np.stack(selected_index)
        cloud_index = np.stack(cloud_index)

        flat_inputs = self.tf_map(selected_pointcloud, selected_labels, selected_index, cloud_index)

        num_layers = Config_SemanticKITTI.number_layers
        inputs = {}

        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())

        inputs['neighbour_index'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neighbour_index'].append(torch.from_numpy(tmp).long())

        inputs['sub_index'] = []
        for tmp in flat_inputs[2 * num_layers : 3 * num_layers]:
            inputs['sub_index'].append(torch.from_numpy(tmp).long())
        
        inputs['interplation_index'] = []
        for tmp in flat_inputs[3 * num_layers : 4 * num_layers]:
            inputs['interplation_index'].append(torch.from_numpy(tmp).long())

        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_index'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_index'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        return inputs