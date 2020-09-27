#! ~/.miniconda3/envs/pytorch/bin/python

class Config_Semantic3D:
    knn = 16
    number_layers = 5
    num_points = 65536
    num_classes = 8
    sub_grid_size = 0.06

    batch_size = 4
    val_batch_size = 16
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    dimension_out = [16, 64, 128, 256, 512]

    noise_init = 3.5
    max_epochs = 100
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'semantic3d_train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8
