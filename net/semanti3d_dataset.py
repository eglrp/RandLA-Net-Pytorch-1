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
from config.config_semantic3d import ConfigSemantic3D

class Semantic3D(torch_data.Dataset):
    def __init__(self):
        self.name = 'Semantic3D'
        self.dataset_path = os.path.join(root_dir, 'data/semantic3d')
        self.label_to_names = {0: 'unlabeled',
                                1: 'man-made terrain',
                                2: 'natural terrain',
                                3: 'high vegetation',
                                4: 'low vegetation',
                                5: 'buildings',
                                6: 'hard scape',
                                7: 'scanning artefacts',
                                8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.original_folder = os.path.join(self.path, 'original_data')
        self.full_pc_folder = os.path.join(self.path, 'original_ply')
        self.sub_pc_folder = os.path.join(self.path, 'input_{:.3f}'.format(ConfigSemantic3D.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 1

        # Initial training-validation-testing files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
        for pc_name in cloud_names:
            if os.path.exists(os.path.join(self.original_folder, pc_name + '.labels')):
                self.train_files.append(os.path.join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                self.test_files.append(os.path.join(self.full_pc_folder, pc_name + '.ply'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, file_path in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {'training': [], 'validation': [], 'test': []}
        self.min_possibility = {'training': [], 'validation': [], 'test': []}
        self.class_weight = {'training': [], 'validation': [], 'test': []}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(ConfigSemantic3D.sub_grid_size)

    
    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))
        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
                self.split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
                self.split = 'training'
            else:
                cloud_split = 'test'
                self.split = 'test'

            # Name of the input files
            kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):
            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = os.path.join(tree_path, '{:s}_project.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = os.path.join(tree_path, '{:s}_project.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        
        for i, tree in enumerate(self.input_trees[self.split]):
            self.possibility[self.split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[self.split] += [float(np.min(self.possibility[self.split][-1]))]

        if self.split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[self.split]), return_counts=True)
            self.class_weight[self.split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

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
        noise = np.random.normal(scale=ConfigSemantic3D.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        query_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=ConfigSemantic3D.num_points)[1][0]
        
        # Shuffle index
        query_idx = DataProcessing.shuffle_idx(query_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]
        queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
        queried_pc_colors = self.input_colors[self.split][cloud_idx][query_idx]
        if self.split == 'test':
            queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
            queried_pt_weight = 1
        else:
            queried_pc_labels = self.input_labels[self.split][cloud_idx][query_idx]
            queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
            queried_pt_weight = np.array([self.class_weight[self.split][0][n] for n in queried_pc_labels])

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
        self.possibility[self.split][cloud_idx][query_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        return queried_pc_xyz, queried_pc_colors.astype(np.float32), queried_pc_labels, query_idx.astype(np.int32), np.array([cloud_idx], dtype=np.int32)

    def tf_augment_input(self, inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = torch.Tensor(1,).uniform_(0, 2 * np.pi)
        c = torch.cos(theta)
        s = torch.sin(theta)
        cs0 = torch.zeros_like(c)
        cs1 = torch.ones_like(c)
        R = torch.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], dim=3)
        stacked_rots = tf.reshape(R, (3, 3))


    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigSemantic3D.num_layers):
            # print('queried_pc_xyz shape:',batch_xyz.shape) # (1, N, 3)
            neighbour_idx = DataProcessing.knn_search(batch_xyz, batch_xyz, ConfigSemantic3D.k_n)
            # print('neighbour_idx shape:', neighbour_idx.shape) # (1, N, 16)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // ConfigSemantic3D.sub_sampling_ratio[i], :]
            # print('sub_points shape:', sub_points.shape) # (1, N, 16)
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // ConfigSemantic3D.sub_sampling_ratio[i], :]
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