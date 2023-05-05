def conf_default():
    return {
        'logname': 'default_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        # "glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64',
        'data_path_test': '/home/francisco/dataset/capitals64/test',
        'use_mask': False,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/',
        'default_font': '/home/francisco/dataset/capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment1_var1():
    return {
        'logname': 'default_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        # "glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64',
        'data_path_test': '/home/francisco/dataset/capitals64/test',
        'use_mask': False,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/',
        'default_font': '/home/francisco/dataset/capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment1_var2():
    return {
        'logname': 'mask_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        # "glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64',
        'data_path_test': '/home/francisco/dataset/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/',
        'default_font': '/home/francisco/dataset/capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment1_var3():
    return {
        'logname': 'skeleton_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        # "glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '../dataset/capitals64',
        'data_path_test': '../dataset/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': True,
        'masks_path': '../dataset/masks/train/',
        'default_font': '../dataset/capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment2_var1():
    return {
        'logname': 'subset_1_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment2',
        'n_visible': 6,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        "glyphs_visible": [10, 7, 1, 13, 17, 3],
        'data_path': '../dataset/capitals64',
        'data_path_test': '../datasets/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '../dataset/masks/train/',
        'default_font': '../capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment2_var2():
    return {
        'logname': 'subset_2_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment2',
        'n_visible': 6,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        "glyphs_visible": [7, 14, 3, 18, 6, 17],
        'data_path': '../capitals64',
        'data_path_test': '../datasets/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '../dataset/masks/train/',
        'default_font': '../dataset/capitals64/BASE/Code New Roman.0.0.png'
    }

def conf_experiment3():
    return {
        'logname': 'default_n_3_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment3',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        "glyphs_visible": None,
        'data_path': '../dataset/capitals64',
        'data_path_test': '../datasets/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '../dataset/masks/train/',
        'default_font': '../datasets/capitals64/BASE/Code New Roman.0.0.png'
    }


def conf_experiment4_var3():
    return {
        'logname': 'Train_Original',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment4',
        'n_visible': 6,
        'epochs': 600,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        "glyphs_visible": [7, 14, 3, 18, 6, 17],
        'data_path': '../dataset/capitals64',
        'data_path_test': '../datasets/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '/dataset/masks/train/',
        'default_font': '../capitals64/BASE/Code New Roman.0.0.png'
    }

# {K,H,B,N,R,D} subset 1
# [10,7,1,13,17,3]

# {H,O,D,S,G,R} subset 2
# [7,14,3,18,6,17]
