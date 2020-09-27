#! ~/.miniconda3/envs/pytorch/bin/python

class Config_S3DIS:
    knn = 16
    num_layers = 5
    num_points = 4096 * 10
    num_classes = 13
    sub_grid_size = 0.04

    batch_size = 6
    val_batch_size = 20
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    dimension_out = [16, 64, 128, 256, 512]

    noise_init = 3.5
    max_epochs = 100
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'semantic_s3dis_train_log'
    saving = True
    saving_path = None
