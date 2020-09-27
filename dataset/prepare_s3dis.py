#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import pprint
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


s3dis_data_path = os.path.join(
    root_dir, 'data/s3dis/Stanford3dDataset_v1.2_Aligned_Version')
annotations_paths = [line.rstrip() for line in open(
    os.path.join(s3dis_data_path, 'annotations.txt'))]
annotations_paths = [os.path.join(s3dis_data_path, path)
                     for path in annotations_paths]
ground_truth_class = [line.rstrip() for line in open(
    os.path.join(s3dis_data_path, 'class_names.txt'))]
ground_truth_label = {class_name: i for i,
                      class_name in enumerate(ground_truth_class)}

sub_grid_size = 0.04
original_pointcloud_folder = os.path.join(os.path.dirname(s3dis_data_path), 'original_ply')
sub_pointcloud_folder = os.path.join(os.path.dirname(s3dis_data_path), 'input_{:.3f}'.format(sub_grid_size))

def make_dir(sub_grid_size=0.04):
    """create dataset path

    Args:
        sub_grid_size (float, optional): [description]. Defaults to 0.04.
    """      
    original_pointcloud_folder = os.path.join(
        os.path.dirname(s3dis_data_path), 'original_ply')
    sub_pointcloud_folder = os.path.join(os.path.dirname(
        s3dis_data_path), 'input_{:.3f}'.format(sub_grid_size))
    os.mkdir(original_pointcloud_folder) if not os.path.exists(
        original_pointcloud_folder) else None
    os.mkdir(sub_pointcloud_folder) if not os.path.exists(
        sub_pointcloud_folder) else None


def convert_pointcloud2ply(annotations_path, save_path, sub_grid_size=0.04):
    """convert original files(.txt) to ply file(each line is XYZRGBL).

    Args:
        annotations_path (str): path to annotations
        save_path (str): path to save original point clouds (each line is XYZRGBL)
        sub_grid_size (float, optional): [description]. Defaults to 0.04.
    """    
    make_dir(sub_grid_size)
    data_list = []
    for file in glob.glob(os.path.join(annotations_path, '*.txt')):
        class_name = os.path.basename(file).split('_')[0]

        if class_name not in ground_truth_class:
            class_name = 'clutter'

        pointcloud = pd.read_csv(
            file, header=None, delim_whitespace=True).values
        labels = np.ones(
            (pointcloud.shape[0], 1)) * ground_truth_label[class_name]
        data = np.concatenate([pointcloud, labels],
                              axis=1)  # x,y,z,r,g,b,label
        data_list.append(data)

    pointcloud_and_label = np.concatenate(
        [data for data in data_list], axis=0)
    xyz_min = np.min(pointcloud_and_label, axis=0)[0:3]
    pointcloud_and_label[:, 0:3] = pointcloud_and_label[:, 0:3] - xyz_min

    xyz = pointcloud_and_label[:, 0:3].astype(np.float32)
    colors = pointcloud_and_label[:, 3:6].astype(np.uint8)
    labels = pointcloud_and_label[:, 6].astype(np.uint8)
    ply.write_ply(save_path, (xyz, colors, labels),  [
                  'x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    sub_xyz, sub_colors, sub_labels = DataProcessing.grid_sub_sampling(
        xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(
        sub_pointcloud_folder, save_path.split('/')[-1][:-4] + '.ply')
    ply.write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], [
                  'x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = os.path.join(sub_pointcloud_folder, str(
        save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    project_index = np.squeeze(search_tree.query(xyz, return_distance=False))
    project_index = project_index.astype(np.int32)
    project_save = os.path.join(sub_pointcloud_folder, str(
        save_path.split('/')[-1][:-4]) + '_project.pkl')
    with open(project_save, 'wb') as f:
        pickle.dump([project_index, labels], f)


if __name__ == '__main__':
    for annotations_path in annotations_paths:
        elements = str(annotations_path).split('/')
        output_filename = elements[-3]+'_'+elements[-2]+'.ply'
        convert_pointcloud2ply(annotations_path, os.path.join(
            original_pointcloud_folder, output_filename))
