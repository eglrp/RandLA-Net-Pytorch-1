#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import open3d
import modin.pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from utils.ply import read_ply
from dataset.dataploting import Plot

semantic3d_orighinal_data_dir = os.path.join(root_dir, 'data/semantic3d/')
semantic3d_pointcloud_list = np.sort(
    glob.glob(
        os.path.join(
            os.path.join(semantic3d_orighinal_data_dir, 'original_ply'),
            '*.ply')))
semantic3d_pointcloud_names = [
    file_name[:-4] for file_name in os.listdir(
        os.path.join(semantic3d_orighinal_data_dir, 'original_ply'))
]
semantic3d_label_list = np.sort(
    glob.glob(
        os.path.join(
            os.path.join(semantic3d_orighinal_data_dir, 'original_data'),
            '*.labels')))
semantic3d_label_names = [
    file_name[:-4] for file_name in os.listdir(
        os.path.join(semantic3d_orighinal_data_dir, 'original_data'))
    if file_name[-4:] == '.txt'
]

plot_colors = Plot.random_colors(11, seed=2)

for file in semantic3d_label_names:
    if file in semantic3d_pointcloud_names:
        data = read_ply(
            os.path.join(
                os.path.join(semantic3d_orighinal_data_dir, 'original_ply'),
                file + '.ply'))
        pc_xyzrgb = np.vstack((data['x'], data['y'], data['z'], data['red'],
                               data['green'], data['blue'])).T
        Plot.draw_pointcloud(pc_xyzrgb, 'pointcloud')

        pc_xyz = np.vstack((data['x'], data['y'], data['z'])).T
        label = pd.read_csv(os.path.join(
            os.path.join(semantic3d_orighinal_data_dir, 'original_data'),
            file + '.txt'),
                            header=None,
                            na_filter=False,
                            delim_whitespace=True).to_numpy().astype(int)
        Plot.draw_pointcloud_semantic_instance(pc_xyz, label,
                                               'pointcloud_label', plot_colors)
