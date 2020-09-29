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
    def load_pointcloud_semantic3d(filename):
        pointcloud = pd.read_csv(
            filename, header=None, delim_whitespace=True, dtype=np.float16)
        pointcloud = pointcloud.values
        return pointcloud

    @staticmethod
    def load_label_semantic3d(filename):
        labels = pd.read_csv(filename, header=None,
                             delim_whitespace=True, dtype=np.uint8)
        labels = labels.values
        return labels

    @staticmethod
    def load_pointcloud_semantickitti(filepath):
        data = np.fromfile(filepath, dtype=np.float32)
        data = data.reshape((-1, 4))
        pointcloud = data[:, 0:3]
        return pointcloud

    @staticmethod
    def load_label_semantickitti(filepath, remap_lut):
        label = np.fromfile(filepath, dtype=np.uint32)
        label = label.reshape((-1))
        semnatic_label = label & 0xFFFF
        instance_label = label >> 16
        assert (semnatic_label + (instance_label << 16) == label).all()
        semnatic_label = remap_lut[semnatic_label]
        return semnatic_label.astype(np.int32)

    @staticmethod
    def get_pointcloud_list_semantickitti(dataset_path):
        seq_list = np.sort(os.listdir(dataset_path))
        file_list = []
        for seq_index in seq_list:
            if(int(seq_index)<11):
                seq_path = os.path.join(dataset_path, seq_index)
                pointcloud_path = os.path.join(seq_path, 'velodyne')
                file_list.append([os.path.join(pointcloud_path, file) for file in np.sort(os.listdir(pointcloud_path))])
        return np.concatenate(file_list, axis=0)

    @staticmethod
    def get_label_list_semantickitti(dataset_path):
        seq_list = np.sort(os.listdir(dataset_path))
        file_list = []
        for seq_index in seq_list:
            if(int(seq_index)<11):
                seq_path = os.path.join(dataset_path, seq_index)
                label_path = os.path.join(seq_path, 'labels')
                file_list.append([os.path.join(label_path, file) for file in np.sort(os.listdir(label_path))])
        return np.concatenate(file_list, axis=0)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        sequence_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []

        for sequence_id in sequence_list:
            sequence_path = os.path.join(dataset_path, sequence_id)
            pointcloud_path = os.path.join(sequence_path, 'velodyne')
            # print(sequence_path)
            # print(pointcloud_path)

            if sequence_id == '08':
                val_file_list.append([os.path.join(pointcloud_path, file) for file in np.sort(os.listdir(pointcloud_path))])
                if sequence_id == test_scan_num:
                    test_file_list.append([os.path.join(pointcloud_path, file) for file in np.sort(os.listdir(pointcloud_path))])
            elif int(sequence_id) >= 11 and sequence_id == test_scan_num:
                test_file_list.append([os.path.join(
                    pointcloud_path, file) for file in np.sort(os.listdir(pointcloud_path))])
            elif sequence_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([os.path.join(
                    pointcloud_path, file) for file in np.sort(os.listdir(pointcloud_path))])

        train_file_list = np.concatenate(
            [train_file for train_file in train_file_list], axis=0)
        val_file_list = np.concatenate(
            [val_file for val_file in val_file_list], axis=0)
        if test_scan_num != 'None':
            test_file_list = np.concatenate(
                [test_file for test_file in test_file_list], axis=0)
        else:
            test_file_list = None

        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(ref, query, k):
        # neighbours_idx = nearest_neighbors.knn_batch(ref, query, k, omp=True)
        # return neighbours_idx.astype(np.int32)

        assert torch.cuda.is_available()
        knn = KNN(k, transpose_mode=True)
        ref_cuda = torch.from_numpy(ref)
        ref_cuda = ref_cuda.cuda()
        query_cuda = torch.from_numpy(query)
        query_cuda = query_cuda.cuda()
        dist, index = knn(ref_cuda, query_cuda)
        return index.cpu().numpy().astype(np.int32)

    @staticmethod
    def data_augment(xyz, color, label, index, num_output):
        num_input = len(xyz)
        duplicates = np.random.choice(num_input, num_output - num_input)

        xyz_duplicates = xyz[duplicates, ...]
        xyz_augments = np.concatenate([xyz, xyz_duplicates], axis=0)

        color_duplicates = color[duplicates, ...]
        color_augments = np.concatenate([color, color_duplicates], axis=0)

        index_duplicates = list(range(num_input)) + list(duplicates)
        index_augments = index[index_duplicates]

        label_augments = label[index_duplicates]

        return xyz_augments, color_augments, index_augments, label_augments

    @staticmethod
    def shuffle_index(data):
        index = np.arange(len(data))
        np.random.shuffle(index)
        return data[index]

    @staticmethod
    def shuffle_list(data):
        indices = np.arange(np.shape(data)[0])
        np.random.shuffle(indices)
        data = data[indices]
        return data

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if features is None and labels is None:
            return grid_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return grid_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return grid_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return grid_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)

    @staticmethod
    def IOU_from_confusions(confusions):
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_Plus_FN = np.sum(confusions, axis=-1)
        TP_Plus_FP = np.sum(confusions, axis=-2)

        IOU = TP / (TP_Plus_FP + TP_Plus_FN - TP + 1e-6)

        mask = TP_Plus_FN < 1e-3
        counts = np.sum(1-mask, axis=-1, keepdims=True)
        mIOU = np.sum(IOU, axis=-1, keepdims=True) / (counts + 1e-6)

        IOU += mask * mIOU
        return IOU

    @staticmethod
    def get_class_weights(dataset_name):
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array(
                [5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353], dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)
