'''
Author: your name
Date: 2020-11-30 09:38:41
LastEditTime: 2020-12-02 09:02:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /RandLA-Net-Pytorch/config/config_semantickitti.py
'''
#! ~/.miniconda3/envs/pytorch/bin/python


class ConfigSemanticKITTI:
    channels = 3
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 28  # batch_size during training
    val_batch_size = 28  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4,
                          4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [
        num_points // 4, num_points // 16, num_points // 64, num_points // 256
    ]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 400  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'semantickitti_train_log'
    saving = True
    saving_path = None
