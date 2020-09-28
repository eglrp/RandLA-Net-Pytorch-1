#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import yaml
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

from utils import ply
from dataset.dataprocessing import DataProcessing

data_config = os.path.join(root_dir, 'data/semantickitti/semantickitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

sub_grid_size = 0.06
dataset_path = os.path.join(root_dir, 'data/semantickitti/dataset/sequences')
output_path = os.path.join(root_dir, 'data/semantickitti/dataset/sequences' + '_' + str(sub_grid_size))
sequence_list = np.sort(os.listdir(dataset_path))

for seq_id in sequence_list:
    print('sequence' + seq_id + ' start')
    seq_path = os.path.join(dataset_path, seq_id)
    seq_path_out = os.path.join(output_path, seq_id)
    pc_path = os.path.join(seq_path, 'velodyne')
    pc_path_out = os.path.join(seq_path_out, 'velodyne')
    KDTree_path_out = os.path.join(seq_path_out, 'KDTree')
    os.makedirs(seq_path_out) if not os.path.exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not os.path.exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not os.path.exists(KDTree_path_out) else None

    if int(seq_id) < 11:
        label_path = os.path.join(seq_path, 'labels')
        label_path_out = os.path.join(seq_path_out, 'labels')
        os.makedirs(label_path_out) if not os.path.exists(label_path_out) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DataProcessing.load_pointcloud_semantickitti(os.path.join(pc_path, scan_id))
            labels = DataProcessing.load_label_semantickitti(os.path.join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
            sub_points, sub_labels = DataProcessing.grid_sub_sampling(points, labels=labels, grid_size=sub_grid_size)
            search_tree = KDTree(sub_points)
            KDTree_save = os.path.join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            np.save(os.path.join(pc_path_out, scan_id)[:-4], sub_points)
            np.save(os.path.join(label_path_out, scan_id)[:-4], sub_labels)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            if seq_id == '08':
                proj_path = os.path.join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not os.path.exists(proj_path) else None
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_save = os.path.join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
    else:
        proj_path = os.path.join(seq_path_out, 'proj')
        os.makedirs(proj_path) if not os.path.exists(proj_path) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DataProcessing.load_pointcloud_semantickitti(os.path.join(pc_path, scan_id))
            sub_points = DataProcessing.grid_sub_sampling(points, grid_size=0.06)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = os.path.join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            proj_save = os.path.join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
            np.save(os.path.join(pc_path_out, scan_id)[:-4], sub_points)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)
