#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
utils_dir = os.path.join(root_dir, 'utils')
sys.path.append(base_dir)
sys.path.append(root_dir)
sys.path.append(utils_dir)

from dataset.dataprocessing import DataProcessing
from utils import ply

modelnet_data_path = os.path.join(root_dir,
                                  'data/modelnet/modelnet40_normal_resampled')
file_paths = [
    line.rstrip()
    for line in open(os.path.join(modelnet_data_path, 'filelist.txt'))
]
file_paths = [os.path.join(modelnet_data_path, path) for path in file_paths]
ground_truth_class = [
    line.rstrip() for line in open(
        os.path.join(modelnet_data_path, 'modelnet40_shape_names.txt'))
]
ground_truth_label = {
    class_name: i
    for i, class_name in enumerate(ground_truth_class)
}
print(ground_truth_label)

sub_grid_size = 0.06
original_pointcloud_folder = os.path.join(os.path.dirname(modelnet_data_path),
                                          'original_ply')
sub_pointcloud_folder = os.path.join(os.path.dirname(modelnet_data_path),
                                     'input_{:.3f}'.format(sub_grid_size))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def make_dir(sub_grid_size=0.04):
    """create dataset path

    Args:
        sub_grid_size (float, optional): [description]. Defaults to 0.04.
    """

    original_pointcloud_folder = os.path.join(
        os.path.dirname(modelnet_data_path), 'original_ply')
    sub_pointcloud_folder = os.path.join(os.path.dirname(modelnet_data_path),
                                         'input_{:.3f}'.format(sub_grid_size))
    os.mkdir(original_pointcloud_folder
             ) if not os.path.exists(original_pointcloud_folder) else None
    os.mkdir(sub_pointcloud_folder
             ) if not os.path.exists(sub_pointcloud_folder) else None


def convert_pointcloud2ply(annotations_path, save_path, sub_grid_size=0.04):
    """convert original files(.txt) to ply file(each line is XYZRGBL).

    Args:
        annotations_path (str): path to annotations
        save_path (str): path to save original point clouds (each line is XYZRGBL)
        sub_grid_size (float, optional): [description]. Defaults to 0.04.
    """
    make_dir(sub_grid_size)

    class_name = os.path.basename(annotations_path).split('/')[0][:-9]
    pointcloud = np.loadtxt(annotations_path, delimiter=',').astype(np.float32)
    labels = np.ones((pointcloud.shape[0], 1)) * ground_truth_label[class_name]
    pointcloud_and_label = np.concatenate([pointcloud, labels], axis=1)
    xyz_min = np.min(pointcloud_and_label, axis=0)[0:3]
    pointcloud_and_label[:, 0:3] = pointcloud_and_label[:, 0:3] - xyz_min

    xyz = pointcloud_and_label[:, 0:3].astype(np.float32)
    colors = pointcloud_and_label[:, 3:6].astype(np.uint8)
    labels = pointcloud_and_label[:, 6].astype(np.uint8)

    print(save_path)
    ply.write_ply(save_path, (xyz, colors, labels),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    sub_xyz, sub_colors, sub_labels = DataProcessing.grid_sub_sampling(
        xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(sub_pointcloud_folder,
                                save_path.split('/')[-1][:-4] + '.ply')
    print(sub_ply_file)
    ply.write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = os.path.join(
        sub_pointcloud_folder,
        str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    print(kd_tree_file)
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    project_index = np.squeeze(search_tree.query(xyz, return_distance=False))
    project_index = project_index.astype(np.int32)
    project_save = os.path.join(
        sub_pointcloud_folder,
        str(save_path.split('/')[-1][:-4]) + '_project.pkl')
    print(project_save)
    with open(project_save, 'wb') as f:
        pickle.dump([project_index, labels], f)


if __name__ == '__main__':
    for file_path in file_paths:
        elements = str(file_path).split('/')
        output_filename = elements[-3] + '_' + elements[-1][:-4] + '.ply'
        convert_pointcloud2ply(
            file_path, os.path.join(original_pointcloud_folder,
                                    output_filename), sub_grid_size)
