#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import glob
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from utils.ply import read_ply
from dataset.dataploting import Plot

s3dis_base_dir = os.path.join(root_dir, 'data/s3dis/results')
s3dis_orighinal_data_dir = os.path.join(root_dir, 'data/s3dis/original_ply')
s3dis_data_path = np.sort(glob.glob(os.path.join(s3dis_base_dir, '*.ply')))

def viewer(original_dir, data_dir, config, visualization = False):
    test_total_correct = 0
    test_total_seen = 0
    gt_classes = [0 for _ in range(config.num_classes)]
    positive_classes = [0 for _ in range(config.num_classes)]
    true_positive_classes = [0 for _ in range(config.num_classes)]
    

    for file_name in data_dir:
        predict_data = read_ply(file_name)
        predict = predict_data['predict']
        
        original_data = read_ply(os.path.join(original_dir, file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

        if visualization:
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            xyzrgb = np.concatenate([points, colors], axis=-1)
            
            Plot.draw_pointcloud(xyzrgb)
            Plot.draw_pointcloud_semantic_instance(points, labels)
            Plot.draw_pointcloud_semantic_instance(points, predict)

        correct = np.sum(predict == labels)
        print(str(file_name.split('/')[-1][:-4]) + '_accuracy:' + str(correct / float(len(labels))))
        
        test_total_correct += correct
        test_total_seen += len(labels)
        
        for j in range(len(labels)):
            gt_label = int(labels[j])
            predict_label = int(predict[j])

            gt_classes[gt_label] += 1
            positive_classes[predict_label] += 1
            true_positive_classes[gt_label] += int(gt_label == predict_label)

        iou_list = []
        for n in range(config.num_classes):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(config.num_classes)

        print('eval_accuracy:{}'.format(test_total_correct / float(test_total_seen)))
        print('mean iou:{}'.format(mean_iou))
        print('iou_list', iou_list)

        accuracy_list = []
        for n in range(config.num_classes):
            accuracy = true_positive_classes[n] / float(gt_classes[n])
            accuracy_list.append(accuracy)
        mean_accuracy = sum(accuracy_list) / float(config.num_classes)
        print('mean accuracy:{}'.format(mean_accuracy))
