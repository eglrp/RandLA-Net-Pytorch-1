#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import torch
import random
import colorsys
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
utils_dir = os.path.join(root_dir, 'utils')
sys.path.append(base_dir)
sys.path.append(root_dir)
sys.path.append(utils_dir)

from utils.nearest_neighbors.lib.python import nearest_neighbors
from utils.cpp_wrappers.cpp_subsampling import grid_subsampling
from knn_cuda import KNN


class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename,
                            header=None,
                            delim_whitespace=True,
                            dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename,
                               header=None,
                               delim_whitespace=True,
                               dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_pointcloud_list_semantickitti(dataset_path):
        seq_list = np.sort(os.listdir(dataset_path))
        file_list = []
        for seq_index in seq_list:
            if (int(seq_index) < 11):
                seq_path = os.path.join(dataset_path, seq_index)
                pointcloud_path = os.path.join(seq_path, 'velodyne')
                file_list.append([
                    os.path.join(pointcloud_path, file)
                    for file in np.sort(os.listdir(pointcloud_path))
                ])
        return np.concatenate(file_list, axis=0)

    @staticmethod
    def get_label_list_semantickitti(dataset_path):
        seq_list = np.sort(os.listdir(dataset_path))
        file_list = []
        for seq_index in seq_list:
            if (int(seq_index) < 11):
                seq_path = os.path.join(dataset_path, seq_index)
                label_path = os.path.join(seq_path, 'labels')
                file_list.append([
                    os.path.join(label_path, file)
                    for file in np.sort(os.listdir(label_path))
                ])
        return np.concatenate(file_list, axis=0)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = os.path.join(dataset_path, seq_id)
            pc_path = os.path.join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([
                    os.path.join(pc_path, f)
                    for f in np.sort(os.listdir(pc_path))
                ])
                if seq_id == test_scan_num:
                    test_file_list.append([
                        os.path.join(pc_path, f)
                        for f in np.sort(os.listdir(pc_path))
                    ])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([
                    os.path.join(pc_path, f)
                    for f in np.sort(os.listdir(pc_path))
                ])
            elif seq_id in [
                    '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]:
                train_file_list.append([
                    os.path.join(pc_path, f)
                    for f in np.sort(os.listdir(pc_path))
                ])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        if test_scan_num != 'None':
            test_file_list = np.concatenate(test_file_list, axis=0)
        else:
            test_file_list = None
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(ref, query, k):
        neighbour_idx = nearest_neighbors.knn_batch(ref, query, k, omp=True)
        return neighbour_idx.astype(np.int32)

        # assert torch.cuda.is_available()
        # knn = KNN(k, transpose_mode=True)
        # ref_cuda = torch.from_numpy(ref)
        # ref_cuda = ref_cuda.cuda()
        # query_cuda = torch.from_numpy(query)
        # query_cuda = query_cuda.cuda()
        # dist, index = knn(ref_cuda, query_cuda)
        # return index.cpu().numpy().astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points,
                          features=None,
                          labels=None,
                          grid_size=0.1,
                          verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return grid_subsampling.compute(points,
                                            sampleDl=grid_size,
                                            verbose=verbose)
        elif labels is None:
            return grid_subsampling.compute(points,
                                            features=features,
                                            sampleDl=grid_size,
                                            verbose=verbose)
        elif features is None:
            return grid_subsampling.compute(points,
                                            classes=labels,
                                            sampleDl=grid_size,
                                            verbose=verbose)
        else:
            return grid_subsampling.compute(points,
                                            features=features,
                                            classes=labels,
                                            sampleDl=grid_size,
                                            verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([
                3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                650464, 791496, 88727, 1284130, 229758, 2272837
            ],
                                     dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([
                5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860,
                269353
            ],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([
                55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
                78858, 240942562, 17294618, 170599734, 6369672, 230413074,
                101130274, 476491114, 9833174, 129609852, 4506626, 1168181
            ])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)
