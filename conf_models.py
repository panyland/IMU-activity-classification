import numpy as np

# 1 = posture (A1, 7 classes), 2 = movement (B1, 5 classes)
experiment_number = 2

if experiment_number == 1:
    label_key = 'A1'
    num_classes = 7
    class_names = ['Supine', 'Standing', 'Sitting', 'SideRight', 'SideLeft', 'Prone', 'CrawlPosture']
    prior_prob = np.array([0.0752, 0.5798, 0.2123, 0.0263, 0.0356, 0.0447, 0.0261])
elif experiment_number == 2:
    label_key = 'B1'
    num_classes = 5
    class_names = ['Locomotion_Periodic', 'Locomotion_Aperiodic', 'NonLocomotion_Still', 'NonLocomotion_Wobbling', 'Transition']
    prior_prob = np.array([0.2221, 0.0791, 0.3312, 0.2562, 0.1112])

result_dir = f'results{experiment_number}'
name_of_log_textfile = f'trainlog{experiment_number}.txt'
print_conf_contents = True

train_model = True
test_model = True

max_epochs = 800
learning_rate = 4e-5
train_sequence_length = 60
batch_size = 1
patience = 100
window_len = 120
hop_len = 60
train_criterion = 'f1'  # 'f1' or 'recall'

num_folds = 13  # LOSO cross-validation (one per subject file)
randomize_order_kfolds = True

use_class_weights = True
loss_name = 'CrossEntropyLoss'
loss_params = {}

optimization_algorithm = 'Adam'
optimization_algorithm_params = {'lr': learning_rate}

use_lr_scheduler = False
lr_scheduler_name = 'ReduceLROnPlateau'
lr_scheduler_params = {'mode': 'max', 'factor': 0.5, 'patience': 30}

use_augmentation = True
aug_p_noise = 0.0
aug_p_dropout = 0.0
aug_p_rotation = 0.4
aug_p_chandropout = 0.3
aug_p_time_warping = 0.0

encoder_model_params = {
    's_channels': 18,
    'input_channels': window_len,
    'latent_channels': 70,
    'output_channels': 140,
    'dropout': 0.3,
}

timeseries_model_params = {
    'input_channels': 140,
    'residual_channels': 64,
    'postproc_channels': 64,
    'output_channels': num_classes,
    'dilations': [1, 2, 4],
    'filter_width': 5,
    'dropout': 0.4,
}

train_val_ratio = 0.8
shuffle_training_data = True
mix_train_val_subjects = True

params_train_dataset = {
    'train_sequence_length': train_sequence_length,
    'train_val_ratio': train_val_ratio,
    'window_len': window_len,
    'hop_len': hop_len,
    'mix_train_val_subjects': mix_train_val_subjects,
    'augment_train_data': use_augmentation,
    'aug_p_noise': aug_p_noise,
    'aug_p_dropout': aug_p_dropout,
    'aug_p_rotation': aug_p_rotation,
    'aug_p_chandropout': aug_p_chandropout,
    'aug_p_time_warping': aug_p_time_warping,
    'label_key': label_key,
}

params_validation_dataset = {
    'train_sequence_length': train_sequence_length,
    'train_val_ratio': train_val_ratio,
    'window_len': window_len,
    'hop_len': hop_len,
    'mix_train_val_subjects': mix_train_val_subjects,
    'label_key': label_key,
}

params_test_dataset = {
    'train_sequence_length': train_sequence_length,
    'window_len': window_len,
    'hop_len': hop_len,
    'label_key': label_key,
}

params_train = {'batch_size': batch_size, 'shuffle': shuffle_training_data, 'drop_last': False}
params_test = {'batch_size': batch_size, 'shuffle': False, 'drop_last': False}
