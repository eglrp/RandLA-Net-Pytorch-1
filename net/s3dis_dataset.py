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

    def __init__(self, mode, test_area_idx=None):
        self.name = 'S3DIS'
        self.mode = mode
        self.path = os.path.join(root_dir, 'data/S3DIS')
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

        self.all_files = glob.glob(os.path.join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(ConfigS3DIS.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(sub_grid_size))

        if self.mode == 'training' or self.mode == 'validation':
            for i, file_path in enumerate(self.all_files):
                t0 = time.time()
                cloud_name = file_path.split('/')[-1][:-4]
                if self.mode in cloud_name:
                    cloud_split = 'validation'
                else:
                    cloud_split = 'training'

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


        elif self.mode == 'testing':
            for i, file_path in enumerate(self.all_files):
                t0 = time.time()
                cloud_name = file_path.split('/')[-1][:-4]

                if self.val_split in cloud_name:
                    proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                    with open(proj_file, 'rb') as f:
                        proj_idx, labels = pickle.load(f)
                    self.val_proj += [proj_idx]
                    self.val_labels += [labels]
                    print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        self.possibility[self.mode] = []
        self.min_possibility[self.mode] = []
        
        if self.mode == 'training' or self.mode == 'validation':
            for i, tree in enumerate(self.input_colors[self.mode]):
                self.possibility[self.mode] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_possibility[self.mode] += [float(np.min(self.possibility[self.mode][-1]))]
    
    def __len__(self):
        pass

    def __getitem__(self, item):
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx = self.spatially_regular_gen(item)
        return queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx

    def spatially_regular_gen(self, item):
        # Generator loop
        cloud_idx = int(np.argmin(self.min_possibility[self.mode]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.mode][cloud_idx])
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.mode][cloud_idx].data, copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=ConfigS3DIS.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < ConfigS3DIS.num_points:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.mode][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.mode][cloud_idx].query(pick_point, k=ConfigS3DIS.num_points)[1][0]

        # Shuffle index
        queried_idx = DataProcessing.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[self.mode][cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[self.mode][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.mode][cloud_idx][queried_idx] += delta
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[self.mode][cloud_idx]))

        # up_sampled with replacement
        if len(points) < ConfigS3DIS.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DataProcessing.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, ConfigS3DIS.num_points)
        
        return queried_pc_xyz.astype(np.float32), queried_pc_colors.astype(np.float32), queried_pc_labels, queried_idx.astype(np.int32)


    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
        batch_features = torch.cat([batch_xyz, batch_features], dim=1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigS3DIS.num_layers):
            neighbour_idx = DataProcessing.knn_search(batch_xyz, batch_xyz, ConfigS3DIS.k_n)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // ConfigS3DIS.sub_sampling_ratio[i], :]
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
    
    def collate_fn(self,batch):
        # todo
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx = [],[],[],[]
        for i in range(len(batch)):
            queried_pc_xyz.append(batch[i][0])
            queried_pc_colors.append(batch[i][1])
            queried_pc_labels.append(batch[i][2])
            queried_idx.append(batch[i][3])

        queried_pc_xyz = np.stack(queried_pc_xyz)
        queried_pc_colors = np.stack(queried_pc_colors)
        queried_pc_labels = np.stack(queried_pc_labels)
        queried_idx = np.stack(queried_idx)

        flat_inputs = self.tf_map(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx)

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
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs