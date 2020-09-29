#! ~/.miniconda3/envs/pytorch/bin/python

class Config_SemanticKITTI:
    knn = 16
    number_layers = 4
    num_points = 4096 * 11
    num_classes = 19
    sub_grid_size = 0.06

    batch_size = 6
    val_batch_size = 20
    train_steps = 500
    val_steps = 500

    sub_sampling_ratio = [4, 4, 4, 4]
    dimension_out = [16, 64, 128, 256]
    num_sub_points = [num_points // 4, num_points //
                      16, num_points // 64, num_points // 256]

    noise_init = 3.5
    max_epoch = 100
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None
