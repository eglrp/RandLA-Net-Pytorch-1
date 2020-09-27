#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import torch
import random
import colorsys
import open3d
import numpy as np
import pandas as pd

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pointcloud(pointcloud_xyzrgb):
        pointcloud = open3d.geometry.PointCloud()
        pointcloud.points = open3d.utility.Vector3dVector(pointcloud_xyzrgb[:, 0:3])

        if pointcloud_xyzrgb.shape[1] == 3:
            open3d.visualization.draw_geometries([pointcloud])

        if np.max(pointcloud_xyzrgb[:, 3:6]) > 20:
            pointcloud.colors = open3d.utility.Vector3dVector(pointcloud_xyzrgb[:, 3:6] / 255.)
        else:
            pointcloud.colors = open3d.utility.Vector3dVector(pointcloud_xyzrgb[:, 3:6])
        
        open3d.visualization.draw_geometries([pointcloud])
        return 0

    @staticmethod
    def draw_pointcloud_semantic_instance(pointcloud_xyz, pointcloud_semantic_instance, plot_colors=None):
        if plot_colors is not None:
            instance_colors = plot_colors
        else:
            instance_colors = Plot.random_colors(len(np.unique(pointcloud_semantic_instance)) + 1, seed=2)
        
        semantic_instance_labels = np.unique(pointcloud_semantic_instance)
        semantic_instance_bbox = []

        Y_colors = np.zeros((pointcloud_semantic_instance.shape[0], 3))
        for index, semantic_instance in enumerate(semantic_instance_labels):
            valid_index = np.argwhere(pointcloud_semantic_instance == semantic_instance)[:, 0]
            
            if semantic_instance <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = instance_colors[semantic_instance]
                else:
                    tp = instance_colors[index]
            
            Y_colors[valid_index] = tp

            valid_xyz = pointcloud_xyz[valid_index]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])

            semantic_instance_bbox.append([[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])
            
        Y_semantic_instance = np.concatenate([pointcloud_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pointcloud(Y_semantic_instance)
        return Y_semantic_instance