#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import time
import glob
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

from utils.ply import read_ply
from dataset.dataprocessing import DataProcessing
from config.config_s3dis import ConfigS3DIS

class S3DIS(torch_data.Dataset):

    def __init__(self,test_area_idx=5):
        self.name = 'S3DIS'
        self.path = os.path.join(root_dir, 'data/s3dis')
        self.label_to_names = {0: 'ceiling',
                                1: 'floor',
                                2: 'wall',
                                3: 'beam',
                                4: 'column',
                                5: 'window',
                                6: 'door',
                                7: 'table',
                                8: 'chair',
                                9: 'sofa',
                                10: 'bookcase',
                                11: 'board',
                                12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(os.path.join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {'training': [], 'validation': []}
        self.min_possibility = {'training': [], 'validation': []}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        ConfigS3DIS.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        ConfigS3DIS.class_weights = DataProcessing.get_class_weights('S3DIS')
        self.load_sub_sampled_clouds(ConfigS3DIS.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
                self.split = 'validation'
            else:
                cloud_split = 'training'
                self.split = 'training'

            # Name of the input files
            kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))
        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = os.path.join(tree_path, '{:s}_project.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
        
        for i, tree in enumerate(self.input_colors[self.split]):
            self.possibility[self.split].append(np.random.rand(tree.data.shape[0]) * 1e-3)  # (0,0.001)
            self.min_possibility[self.split].append(float(np.min(self.possibility[self.split][-1])))

    def __len__(self):
        if self.split == 'training':
            return len(self.input_trees['training']) * 240
        elif self.split == 'validation':
            return len(self.input_trees['validation']) * 240

    def __getitem__(self, item):
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx = self.spatially_regular_gen(item)
        return queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx

    def spatially_regular_gen(self, item):
        # Generator loop
        cloud_idx = int(np.argmin(self.min_possibility[self.split]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.split][cloud_idx].data, copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=ConfigS3DIS.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < ConfigS3DIS.num_points:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=ConfigS3DIS.num_points)[1][0]

        # Shuffle index
        queried_idx = DataProcessing.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[self.split][cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[self.split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.split][cloud_idx][queried_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        # up_sampled with replacement
        if len(points) < ConfigS3DIS.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DataProcessing.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, ConfigS3DIS.num_points)
        
        return queried_pc_xyz.astype(np.float32), queried_pc_colors.astype(np.float32), queried_pc_labels, queried_idx.astype(np.int32), np.array([cloud_idx], dtype=np.int32)


    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigS3DIS.num_layers):
            # print('queried_pc_xyz shape:',batch_xyz.shape) # (1, N, 3)
            neighbour_idx = DataProcessing.knn_search(batch_xyz, batch_xyz, ConfigS3DIS.k_n)
            # print('neighbour_idx shape:', neighbour_idx.shape) # (1, N, 16)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // ConfigS3DIS.sub_sampling_ratio[i], :]
            # print('sub_points shape:', sub_points.shape) # (1, N, 16)
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // ConfigS3DIS.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_xyz, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
        return input_list
    
    def collate_fn(self, batch):
        pass
        # todo
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx = [], [], [], [], []
        for i in range(len(batch)):
            queried_pc_xyz.append(batch[i][0])
            queried_pc_colors.append(batch[i][1])
            queried_pc_labels.append(batch[i][2])
            queried_idx.append(batch[i][3])
            queried_cloud_idx.append(batch[i][4])

        queried_pc_xyz = np.stack(queried_pc_xyz)
        queried_pc_colors = np.stack(queried_pc_colors)
        queried_pc_labels = np.stack(queried_pc_labels)
        queried_idx = np.stack(queried_idx)
        queried_cloud_idx = np.stack(queried_cloud_idx)

        flat_inputs = self.tf_map(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx)

        num_layers = ConfigS3DIS.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        return inputs