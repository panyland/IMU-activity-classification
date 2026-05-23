# 1 for posture, 2 for movement classes
experiment_number = 2

# The .py configuration file that was used for PFML pre-training (contains all necessary model
# hyperparameters). Note: Do NOT add the '.py' ending to the file name!
pretrained_model_conf_file = 'conf_pfml_pretrain_imu'

# The name of the time-series model that we want to use from the file pfml_model.py for fine-tuning
timeseries_model = 'WaveNet'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True


# The directory where the experimental results are saved (the models and the logging output)
result_dir = f'results{experiment_number}'
# The name of the text file into which we log the output of the training process. Please note that this
# file (and its directory) will be saved under the directory result_dir.
name_of_log_textfile = f'trainlog{experiment_number}.txt'


"""The hyperparameters for our training process"""

# Flag for training our model(s)
train_model = True

# Flag for testing our trained model(s)
test_model = True

# Flag for fine-tuning the Transformer model block by block. If set to False, the entire Transformer model
# will be fine-tuned. If set to True, the Transformer model will be fine-tuned N times, where N is the
# number of Transformer encoder blocks in the model. In each fine-tuning iteration (except the first),
# one Transformer encoder block will be removed from the end of the model, and the rest of the model will
# be fine-tuned. For example, if N = 3, this means that we fine-tune the model three times: First, we
# fine-tune the entire model. Then, we remove one Transformer encoder block and we fine-tune the Transformer
# consisting of 2 encoder blocks. Finally, we remove another block, and we fine-tune a Transformer consisting
# of only one encoder block.
finetune_transformer_block_by_block = False

# The maximum number of training epochs
max_epochs = 800

# The learning rate of our model training
learning_rate = 4e-5

# The number of frames in each input sequence for our model (Fs=52 Hz, 60-sample hop length --> 260 frames is 5 minutes)
train_sequence_length = 260

# The number of input sequences that we feed into our model before computing the mean loss (and performing backpropagation
# during training).
batch_size = 1

# The patience counter for early stopping
patience = 100

# Dropout rate of the encoder model
dropout_encoder_model = 0.3

# Dropout rate of the timeseries model
dropout_timeseries_model = 0.4

# Window length (in samples)
window_len = 120

# Hop length (in samples)
hop_len = 60

# Select the training criterion
train_criterion = 'f1' # Options: 'f1' / 'recall'

# The number of folds for k-folds cross-validation
num_folds = 13 # Tiedostojen määrä -> LOSO-cross-validation

# A flag whether we want to randomize the order of the babies before applying k-folds cross-validation
randomize_order_kfolds = True

# A flag to whether we want to use class weighting for our loss
use_class_weights = True

# Define our loss function that we want to use from torch.nn
loss_name = 'CrossEntropyLoss'

# The hyperparameters for the loss function
loss_params = {}

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = False

# Define our learning rate schedulers for the fine-tuning stages 1 and 2 (from torch.optim.lr_scheduler)
lr_scheduler_stage_1 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_1 = {'mode': 'max',
                               'factor': 0.5,
                               'patience': 30}

lr_scheduler_stage_2_part_1_epochs = 20
lr_scheduler_stage_2_part_1 = 'LinearLR'
lr_scheduler_params_stage_2_part_1 = {'start_factor': 0.001,
                                      'total_iters': lr_scheduler_stage_2_part_1_epochs}
lr_scheduler_stage_2_part_2 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_2_part_2 = {'mode': 'max',
                                      'factor': 0.5,
                                      'patience': 30}


"""The hyperparameters for training data augmentation"""

# Select whether we want to use data augmentation for our training data or not
use_augmentation = True

# Probability for additive noise augmentation
aug_p_noise = 0.0

# If we perform additive noise augmentation, the probability for adding noise to samples
aug_p_dropout = 0.0

# Probability for performing a random rotation
aug_p_rotation = 0.0

# Probability for sensor dropout
aug_p_chandropout = 0.3

# Probability for time warping
aug_p_time_warping = 0.0


"""Additional hyperparameters for our models"""
additional_hyperparameters_timeseries_model = {}


"""The hyperparameters for our dataset and data loaders"""

# Define our dataset for our data loader that we want to use from the file pfml_data_loader.py
dataset_name = 'imu_dataset'

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the validation set.
train_val_ratio = 0.8

# Select whether we want to shuffle our training data
shuffle_training_data = True

# Select if we want to split our training and validation data so that baby-specific data is included
# in both sets.
mix_train_val_babies = True

# The hyperparameters for our data loaders
params_train_dataset = {'train_sequence_length': train_sequence_length,
                        'train_val_ratio': train_val_ratio,
                        'window_len': window_len,
                        'hop_len': hop_len,
                        'mix_train_val_babies': mix_train_val_babies,
                        'augment_train_data': use_augmentation,
                        'aug_p_noise': aug_p_noise,
                        'aug_p_dropout': aug_p_dropout,
                        'aug_p_rotation': aug_p_rotation,
                        'aug_p_chandropout': aug_p_chandropout,
                        'aug_p_time_warping': aug_p_time_warping,
                        'include_artificial_labels': True}

params_validation_dataset = {'train_sequence_length': train_sequence_length,
                             'train_val_ratio': train_val_ratio,
                             'window_len': window_len,
                             'hop_len': hop_len,
                             'mix_train_val_babies': mix_train_val_babies,
                             'include_artificial_labels': True}

params_test_dataset = {'train_sequence_length': train_sequence_length,
                       'window_len': window_len,
                       'hop_len': hop_len,
                       'include_artificial_labels': True}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for using our trained data2vec model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': False}
