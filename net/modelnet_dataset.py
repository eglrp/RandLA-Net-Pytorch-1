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
from config.config_modelnet import ConfigMODELNET


class MODELNET(torch_data.Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.path = os.path.join(root_dir, 'data/modelnet')
        self.label_to_names = {
            0: 'airplane',
            1: 'bathtub',
            2: 'bed',
            3: 'bench',
            4: 'bookshelf',
            5: 'bottle',
            6: 'bowl',
            7: 'car',
            8: 'chair',
            9: 'cone',
            10: 'cup',
            11: 'curtain',
            12: 'desk',
            13: 'door',
            14: 'dresser',
            15: 'flower_pot',
            16: 'glass_box',
            17: 'guitar',
            18: 'keyboard',
            19: 'lamp',
            20: 'laptop',
            21: 'mantel',
            22: 'monitor',
            23: 'night_stand',
            24: 'person',
            25: 'piano',
            26: 'plant',
            27: 'radio',
            28: 'range_hood',
            29: 'sink',
            30: 'sofa',
            31: 'stairs',
            32: 'stool',
            33: 'table',
            34: 'tent',
            35: 'toilet',
            36: 'tv_stand',
            37: 'vase',
            38: 'wardrobe',
            39: 'xbox'
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.possibility = {'train': [], 'test': []}
        self.min_possibility = {'train': [], 'test': []}
        self.input_trees = {'train': [], 'test': []}
        self.input_colors = {'train': [], 'test': []}
        self.input_labels = {'train': [], 'test': []}
        self.input_names = {'train': [], 'test': []}

        self.cat = [
            line.rstrip() for line in open(
                os.path.join(
                    self.path,
                    'modelnet40_normal_resampled/modelnet40_shape_names.txt'))
        ]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.shape_ids = {}
        self.shape_ids['train'] = [
            line.rstrip() for line in open(
                os.path.join(
                    self.path,
                    'modelnet40_normal_resampled/modelnet40_train.txt'))
        ]
        self.shape_ids['test'] = [
            line.rstrip() for line in open(
                os.path.join(
                    self.path,
                    'modelnet40_normal_resampled/modelnet40_test.txt'))
        ]

        assert (mode == 'train' or mode == 'test')
        self.shape_names = [
            '_'.join(x.split('_')[0:-1]) for x in self.shape_ids[mode]
        ]
        self.datapath = [
            (self.shape_names[i],
             os.path.join(self.path, 'modelnet40_normal_resampled',
                          self.shape_names[i], self.shape_ids[mode][i]) +
             '.txt') for i in range(len(self.shape_ids[mode]))
        ]
        print('The size of %s data is %d' % (mode, len(self.datapath)))
        # print(
        #     os.path.join(
        #         self.path, 'original_ply', 'modelnet40_normal_resampled_' +
        #         self.shape_ids[mode][0] + '.ply'))
        # self.all_files = glob.glob(
        #     os.path.join(self.path, 'original_ply', '*.ply'))
        self.all_files = [
            os.path.join(
                self.path, 'original_ply', 'modelnet40_normal_resampled_' +
                self.shape_ids[mode][i] + '.ply')
            for i in range(len(self.shape_ids[mode]))
        ]
        self.load_sub_sampled_clouds(0.06, mode)

    def load_sub_sampled_clouds(self, sub_grid_size, mode):
        tree_path = os.path.join(self.path,
                                 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            print(i)
            cloud_name = file_path.split('/')[-1][:-4]
            kd_tree_file = os.path.join(tree_path,
                                        '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path,
                                        '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack(
                (data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[mode] += [search_tree]
            self.input_colors[mode] += [sub_colors]
            self.input_labels[mode] += [sub_labels]
            self.input_names[mode] += [cloud_name]

        for i, tree in enumerate(self.input_colors[mode]):
            self.possibility[mode].append(
                np.random.rand(tree.data.shape[0]) * 1e-3)  # (0,0.001)
            self.min_possibility[mode].append(
                float(np.min(self.possibility[mode][-1])))

    def spatially_regular_gen(self, item):
        # Generator loop
        cloud_idx = int(np.argmin(self.min_possibility[self.mode]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.mode][cloud_idx])
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.mode][cloud_idx].data,
                          copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < ConfigMODELNET.num_points:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.mode][cloud_idx].query(
                pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.mode][cloud_idx].query(
                pick_point, k=40960)[1][0]

        # Shuffle index
        queried_idx = DataProcessing.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[
            self.mode][cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[
            self.mode][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square(
            (points[queried_idx] - pick_point).astype(np.float32)),
                       axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.mode][cloud_idx][queried_idx] += delta
        self.min_possibility[self.mode][cloud_idx] = float(
            np.min(self.possibility[self.mode][cloud_idx]))

        # up_sampled with replacement
        if len(points) < ConfigMODELNET.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DataProcessing.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, ConfigMODELNET.num_points)

        return queried_pc_xyz.astype(np.float32), queried_pc_colors.astype(
            np.float32), queried_pc_labels, queried_idx.astype(
                np.int32), np.array([cloud_idx], dtype=np.int32)

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx,
               batch_cloud_idx, cls):
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigMODELNET.num_layers):
            # print('queried_pc_xyz shape:',batch_xyz.shape) # (1, N, 3)
            neighbour_idx = DataProcessing.knn_search(batch_xyz, batch_xyz,
                                                      ConfigMODELNET.k_n)
            # print('neighbour_idx shape:', neighbour_idx.shape) # (1, N, 16)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] //
                                   ConfigMODELNET.sub_sampling_ratio[i], :]
            # print('sub_points shape:', sub_points.shape) # (1, N, 16)
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] //
                                   ConfigMODELNET.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_xyz, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [
            batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, cls
        ]
        return input_list

    def collate_fn(self, batch):
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx = [], [], [], [], []
        cls = []
        for i in range(len(batch)):
            queried_pc_xyz.append(batch[i][0])
            queried_pc_colors.append(batch[i][1])
            queried_pc_labels.append(batch[i][2])
            queried_idx.append(batch[i][3])
            queried_cloud_idx.append(batch[i][4])
            cls.append(batch[i][5])

        queried_pc_xyz = np.stack(queried_pc_xyz)
        queried_pc_colors = np.stack(queried_pc_colors)
        queried_pc_labels = np.stack(queried_pc_labels)
        queried_idx = np.stack(queried_idx)
        queried_cloud_idx = np.stack(queried_cloud_idx)
        cls = np.stack(cls)

        flat_inputs = self.tf_map(queried_pc_xyz, queried_pc_colors,
                                  queried_pc_labels, queried_idx,
                                  queried_cloud_idx, cls)

        num_layers = ConfigMODELNET.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers:2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(
            flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers +
                                                        1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers +
                                                            2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers +
                                                            3]).long()
        inputs['cls'] = torch.from_numpy(flat_inputs[-1]).long()
        return inputs

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, item):
        cls = self.classes[self.datapath[item][0]]
        # print(cls)
        # cls = np.array([cls]).astype(np.int32)
        queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx = self.spatially_regular_gen(
            item)
        return queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, queried_cloud_idx, cls


s = MODELNET('train')
