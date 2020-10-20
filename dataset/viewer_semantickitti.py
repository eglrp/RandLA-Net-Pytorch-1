#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import yaml
import open3d
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from utils.ply import read_ply
from dataset.dataprocessing import DataProcessing
from dataset.dataploting import Plot

semantickitti_orighinal_data_dir = os.path.join(root_dir, 'data/semantickitti/dataset/sequences')
data_config = os.path.join(root_dir, 'data/semantickitti/semantickitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

def viewer(data_path):
    pointcloud_list = DataProcessing.get_pointcloud_list_semantickitti(data_path)
    label_list = DataProcessing.get_label_list_semantickitti(data_path)

    colors = Plot.random_colors(21, seed=2)

    for index in range(len(pointcloud_list)):
        pointcloud_path = pointcloud_list[index]
        label_path = label_list[index]

        pointcloud = DataProcessing.load_pc_kitti(pointcloud_path)
        label = DataProcessing.load_label_kitti(label_path, remap_lut)
        
        pointcloud_withlabel = np.zeros((pointcloud.shape[0], 6), dtype=np.int)
        pointcloud_withlabel[:,0:3] = pointcloud
        pointcloud_withlabel[:,5] = 1
        Plot.draw_pointcloud(pointcloud, "pointcloud:{}".format(index))
        Plot.draw_pointcloud_semantic_instance(pointcloud, label, "pointcloud_label:{}".format(index))

if __name__ == '__main__':
    viewer(semantickitti_orighinal_data_dir)    
        

