#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import pickle
import pretty_errors
import numpy as np
import pandas as dp
from sklearn.neighbors import KDTree

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
utils_dir = os.path.join(root_dir, 'utils')
sys.path.append(base_dir)
sys.path.append(root_dir)
sys.path.append(utils_dir)

from utils import ply
from dataset.dataprocessing import DataProcessing

semantic3d_data_path = os.path.join(root_dir, 'data/semantic3d/original_data')
sub_grid_size = 0.06
original_pointcloud_folder = os.path.join(
    os.path.dirname(semantic3d_data_path), 'original_ply')
sub_pointcloud_folder = os.path.join(os.path.dirname(semantic3d_data_path),
                                     'input_{:.3f}'.format(sub_grid_size))


def make_dir(sub_grid_size=0.06):
    """create dataset path

    Args:
        sub_grid_size (float, optional): [description]. Defaults to 0.06.
    """
    original_pointcloud_folder = os.path.join(
        os.path.dirname(semantic3d_data_path), 'original_ply')
    subgrid_pointcloud_folder = os.path.join(
        os.path.dirname(semantic3d_data_path),
        'input_{:.3f}'.format(sub_grid_size))
    os.mkdir(original_pointcloud_folder
             ) if not os.path.exists(original_pointcloud_folder) else None
    os.mkdir(subgrid_pointcloud_folder
             ) if not os.path.exists(subgrid_pointcloud_folder) else None


def convert_txt2ply(save_path=None, sub_grid_size=0.06):
    """convert original files to ply file(each line is XYZRGBL).

    Args:
        save_path ([type], optional): [description]. Defaults to None.
        sub_grid_size (float, optional): [description]. Defaults to 0.06.
    """
    make_dir(sub_grid_size)

    for pointcloud_path in glob.glob(
            os.path.join(semantic3d_data_path, '*.txt')):
        print(pointcloud_path)
        filename = pointcloud_path.split('/')[-1][:-4]

        if os.path.exists(
                os.path.join(sub_pointcloud_folder, filename + '_KDTree.pkl')):
            continue

        pointcloud = DataProcessing.load_pc_semantic3d(pointcloud_path)
        label_path = pointcloud_path[:-4] + '.labels'
        print(label_path)
        if os.path.exists(label_path):
            labels = DataProcessing.load_label_semantic3d(label_path)
            full_ply_path = os.path.join(original_pointcloud_folder,
                                         filename + '.ply')

            sub_points, sub_colors, sub_labels = DataProcessing.grid_sub_sampling(
                pointcloud[:, :3].astype(np.float32),
                pointcloud[:, 4:7].astype(np.uint8), labels, 0.01)
            sub_labels = np.squeeze(sub_labels)
            ply.write_ply(full_ply_path, (sub_points, sub_colors, sub_labels),
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            sub_xyz, sub_colors, sub_labels = DataProcessing.grid_sub_sampling(
                sub_points, sub_colors, sub_labels, sub_grid_size)
            sub_colors = sub_colors / 255.0
            sub_labels = np.squeeze(sub_labels)
            sub_ply_file = os.path.join(sub_pointcloud_folder,
                                        filename + '.ply')
            ply.write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            search_tree = KDTree(sub_xyz, leaf_size=50)
            kd_tree_file = os.path.join(sub_pointcloud_folder,
                                        filename + '_KDTree.pkl')
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(search_tree, f)

            proj_idx = np.squeeze(
                search_tree.query(sub_points, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = os.path.join(sub_pointcloud_folder,
                                     filename + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)

        else:
            fully_ply_path = os.path.join(original_pointcloud_folder,
                                          filename + '.ply')
            ply.write_ply(fully_ply_path, (pointcloud[:, :3].astype(
                np.float32), pointcloud[:, 4:7].astype(np.uint8)),
                          ['x', 'y', 'z', 'red', 'green', 'blue'])

            sub_xyz, sub_colors = DataProcessing.grid_sub_sampling(
                pointcloud[:, :3].astype(np.float32),
                pointcloud[:, 4:7].astype(np.uint8),
                grid_size=sub_grid_size)
            sub_colors = sub_colors / 255.0
            sub_ply_file = os.path.join(sub_pointcloud_folder,
                                        filename + '.ply')
            ply.write_ply(sub_ply_file, [sub_xyz, sub_colors],
                          ['x', 'y', 'z', 'red', 'green', 'blue'])
            labels = np.zeros(pointcloud.shape[0], dtype=np.uint8)

            search_tree = KDTree(sub_xyz, leaf_size=50)
            kd_tree_file = os.path.join(sub_pointcloud_folder,
                                        filename + '_KDTree.pkl')
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(search_tree, f)

            proj_idx = np.squeeze(
                search_tree.query(pointcloud[:, :3].astype(np.float32),
                                  return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = os.path.join(sub_pointcloud_folder,
                                     filename + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':
    convert_txt2ply()
