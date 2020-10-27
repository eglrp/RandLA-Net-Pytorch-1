#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import open3d
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from utils.ply import read_ply
from dataset.dataploting import Plot

s3dis_orighinal_data_dir = os.path.join(root_dir, 'data/s3dis/original_ply')
s3dis_data_path = np.sort(
    glob.glob(os.path.join(s3dis_orighinal_data_dir, '*.ply')))


def viewer(original_dir=None, config=None):

    for file_name in s3dis_data_path:
        print(file_name)
        original_data = read_ply(
            os.path.join(s3dis_orighinal_data_dir,
                         file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack(
            (original_data['x'], original_data['y'], original_data['z'])).T

        colors = np.vstack((original_data['red'], original_data['green'],
                            original_data['blue'])).T
        xyzrgb = np.concatenate([points, colors], axis=-1)

        Plot.draw_pointcloud(xyzrgb, "pointcloud")
        Plot.draw_pointcloud_semantic_instance(points, labels,
                                               "pointcloud_label")


if __name__ == '__main__':
    viewer()
