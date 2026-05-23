import numpy as np
import os
import time
import sys

from pathlib import Path
from scipy.io import loadmat
from importlib.machinery import SourceFileLoader
from py_conf_file_into_text import convert_py_conf_file_to_text
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from copy import deepcopy

from torch import cuda, no_grad, save, load, stack, from_numpy
from torch.utils.data import DataLoader
from torch.nn import Softmax


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python finetune_pfml_pretrained_imu_models.py \nOR \n2) python finetune_pfml_pretrained_imu_models.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_models as conf
        conf_file_name = 'conf_models.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python finetune_pfml_pretrained_imu_models.py \nOR \n2) python finetune_pfml_pretrained_imu_models.py <configuration_file>\n\n' \
                 'By using the first option, you need to have a configuration file named "conf_models.py" in the same ' \
                 'directory as "finetune_pfml_pretrained_imu_models.py"')

# Import our timeseries model
timeseries_model = getattr(__import__('models', fromlist=[conf.timeseries_model]), conf.timeseries_model)

# Import our dataset for our data loader
dataset = getattr(__import__('data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
imu_loss = getattr(__import__('torch.nn', fromlist=[conf.loss_name]), conf.loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)


def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed


if __name__ == "__main__":
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(f'{conf.result_dir}/{conf.name_of_log_textfile}')
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    #device = 'cpu' # For debugging
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # Read data
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Generating data...\n')
     

    # Load data
    Data = [] 
    mat_folder = Path("./annotated_IMU_data/")
    mat_files = list(mat_folder.glob("*.mat"))
    for f in mat_files: 
        data = loadmat(f)
        acc = data["acc_data"]
        gyro = data["gyro_data"]  
        X = np.concatenate((acc, gyro), axis=1)
        X_framed = frame_sig(X, conf.window_len, conf.hop_len)
        B1 = data["B1"]
        A1 = data["A1"]
        Nframes = X_framed.shape[0]
        
        B1_framed = np.zeros((Nframes, B1.shape[1]), dtype=B1.dtype)
        for i in range(Nframes):
            center = i * conf.hop_len + conf.window_len // 2
            B1_framed[i] = B1[center]

        A1_framed = np.zeros((Nframes, A1.shape[1]), dtype=A1.dtype)
        for i in range(Nframes):
            center = i * conf.hop_len + conf.window_len // 2
            A1_framed[i] = A1[center]
        
        # Pad short recordings to minimum sequence length
        train_seq_len = conf.params_train_dataset.get('train_sequence_length', 260)
        if X_framed.shape[0] < train_seq_len:
            pad = train_seq_len - X_framed.shape[0]
            X_framed = np.pad(X_framed, ((0, pad), (0, 0), (0, 0)))
            B1_framed = np.pad(B1_framed, ((0, pad), (0, 0)))
            A1_framed = np.pad(A1_framed, ((0, pad), (0, 0)))
            mask = np.ones(train_seq_len)
            mask[:Nframes] = 0
            Nframes = train_seq_len
        else:
            mask = np.zeros(Nframes)
        
        data['X'] = X_framed
        data['B1'] = B1_framed
        data['A1'] = A1_framed
        data['Mask'] = mask
        Data.append(data)
    

    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Done!\n\n')
    
    # We generate a vector of class prior probabilities for the data i.e. empirical distribution from the data
    prior_prob = np.array([0.0752, 0.5798, 0.2123, 0.0263, 0.0356, 0.0447, 0.0261]) # Posture
    #prior_prob = np.array([0.2221, 0.0791, 0.3312, 0.2562, 0.1112]) # Movement
    
    # Compute prior probabilities for labels for (potential) loss weighting
    class_weights = from_numpy(1 / prior_prob).to(device).float()
    
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write(f'\n\n\n\n%%%%%%%%%%%%%%%%%% Fine-tuning the PFML pre-trained model that used the configuration file {conf.pretrained_model_conf_file} %%%%%%%%%%%%%%%%%%\n\n\n\n')
    
    # If the pre-training configuration file is within a folder or folders, we modify
    # the path string to a suitable format
    for separator in ['/', '\\']:
        if separator in conf.pretrained_model_conf_file:
            conf.pretrained_model_conf_file = conf.pretrained_model_conf_file.replace(separator, '.')
    
    conf_pt = __import__(f'{conf.pretrained_model_conf_file}', fromlist=[''])
    
    # We remove pre-training related hyperparameters from the time-series model dict
    removed_items_for_timeseries_model_params_dict = ['require_same_num_embedding_masks',
                                                      'prob_frame_is_start_of_embedding_mask',
                                                      'embedding_mask_length_frames',
                                                      'min_num_mask_start_frames',
                                                      'learnable_mask_embedding',
                                                      'mask_type']
    for removed_item in removed_items_for_timeseries_model_params_dict:
        del conf_pt.transformer_params[removed_item]
    
    encoder_model = getattr(__import__('pfml_model', fromlist=[conf_pt.encoder_name]), conf_pt.encoder_name)
    
    conf_pt.encoder_model_params['dropout'] = conf.dropout_encoder_model
    conf_pt.transformer_params['dropout'] = conf.dropout_timeseries_model
    
    if len(conf.additional_hyperparameters_timeseries_model) != 0:
        for item in conf.additional_hyperparameters_timeseries_model:
            conf_pt.transformer_params[item] = conf.additional_hyperparameters_timeseries_model[item]
    
    if conf.finetune_transformer_block_by_block:
        block_by_block_finetuning_iterations = conf_pt.num_transformer_encoder_layers
    else:
        block_by_block_finetuning_iterations = 1
    
    for num_removed_transformer_encoder_layers in range(block_by_block_finetuning_iterations):
        num_transformer_encoder_layers = conf_pt.num_transformer_encoder_layers - num_removed_transformer_encoder_layers
        conf_pt.transformer_params['num_encoder_layers'] = num_transformer_encoder_layers
        if num_transformer_encoder_layers != conf_pt.num_transformer_encoder_layers:
            if conf_pt.use_final_projection:
                conf_pt.transformer_params['use_final_projection'] = False
    
        best_model_encoder_name_finetuned = f'pfml_imu_best_finetuned_encoder_experiment_{conf.finetuning_experiment_number}_{num_transformer_encoder_layers}_layers.pt'
        best_model_timeseries_name_finetuned = f'pfml_imu_best_finetuned_transformer_experiment_{conf.finetuning_experiment_number}_{num_transformer_encoder_layers}_layers.pt'
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'################## Training using {num_transformer_encoder_layers} Transformer encoder layers ##################\n\n')

        Nbabies = len(Data)
        num_babies_per_fold = int(np.floor(Nbabies // conf.num_folds))
    
        if conf.randomize_order_kfolds:
            perm = np.random.RandomState(seed=42).permutation(Nbabies)
        else:
            perm = np.arange(Nbabies)
        
        fold_test_accuracies = []
        fold_conf_mats = []
        for fold in range(conf.num_folds):
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'Training model for fold {fold+1}\n\n')
            
            if fold == conf.num_folds - 1:
                test_inds = perm[fold*num_babies_per_fold:]
            else:
                test_inds = perm[fold*num_babies_per_fold:(fold+1)*num_babies_per_fold]
    
            D_train = []
            D_test = []
        
            # Select test subject(s)
            for i in range(len(Data)):
                if i in test_inds:
                    D_test.append(Data[i])
                else:
                    D_train.append(Data[i])
    
            conf_pt.transformer_params['output_channels'] = 7 # Number of output categories
            
            # Initialize the data loaders
            if conf.train_model:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Initializing training set...\n')
                training_set = dataset(D_train, train_val_test='train', **conf.params_train_dataset)
                train_data_loader = DataLoader(training_set, **conf.params_train)
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Done!\n')
                    f.write('Initializing validation set...\n')
                validation_set = dataset(D_train, train_val_test='validation', **conf.params_validation_dataset) 
                validation_data_loader = DataLoader(validation_set, **conf.params_train)
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Done!\n')
            if conf.test_model:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Initializing test set...\n')
                test_set = dataset(D_test, train_val_test='test', **conf.params_test_dataset)
                test_data_loader = DataLoader(test_set, **conf.params_test)
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Done!\n\n')
            
            for finetuning_stage in [1, 2]:
                
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'######### Fine-tuning stage {finetuning_stage} #########\n\n')
            
                # Initialize our models, pass the models to the available device
                Encoder_model = encoder_model(**conf_pt.encoder_model_params).to(device)
                Timeseries_model = timeseries_model(**conf_pt.transformer_params).to(device)
                
                # Give the parameters of our models to an optimizer
                model_parameters = list(Encoder_model.parameters()) + list(Timeseries_model.parameters())
                optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
                
                # Get our learning rate for later use
                learning_rate = optimizer.param_groups[0]['lr']
                
                # Give the optimizer to the learning rate scheduler
                if conf.use_lr_scheduler:
                    if finetuning_stage == 1:
                        scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_1]), conf.lr_scheduler_stage_1)
                        lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params_stage_1)
                    else:
                        scheduler_1 = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_2_part_1]), conf.lr_scheduler_stage_2_part_1)
                        scheduler_2 = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_2_part_2]), conf.lr_scheduler_stage_2_part_2)
                        scheduler_part_1 = scheduler_1(optimizer, **conf.lr_scheduler_params_stage_2_part_1)
                        scheduler_part_2 = scheduler_2(optimizer, **conf.lr_scheduler_params_stage_2_part_2)
            
                # Initialize our loss function as a class
                if conf.use_class_weights:
                    loss_function = imu_loss(weight=class_weights, **conf.loss_params)
                else:
                    loss_function = imu_loss(**conf.loss_params)
            
                # Variables for early stopping
                highest_validation_accuracy = -1e10
                best_validation_epoch = 0
                patience_counter = 0
                
                # Load our model weights
                try:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Loading models from file...\n')
                        if finetuning_stage == 1:
                            f.write(f'Loading model {conf_pt.encoder_best_model_name}...\n')
                            f.write(f'Loading model {conf_pt.transformer_best_model_name}...\n')
                            Encoder_model.load_state_dict(load(f'{conf_pt.encoder_best_model_name}', map_location=device))
                            
                            # The pre-trained model does not have classification layers at the end of the network so
                            # we need to filter those out from the parameter dict before we can load the weights. We also filter
                            # out N Transformer encoder blocks if necessary.
                            timeseries_model_state_dict = load(f'{conf_pt.transformer_best_model_name}', map_location=device)
                            filtered_dict = {k: v for k, v in timeseries_model_state_dict.items() if 'classification' not in k}
                            if num_transformer_encoder_layers != conf_pt.num_transformer_encoder_layers:
                                if conf_pt.use_final_projection:
                                    filtered_dict = {k: v for k, v in filtered_dict.items() if 'final_projection' not in k}
                                for i in range(num_removed_transformer_encoder_layers):
                                    removed_transformer_layer_index = conf_pt.num_transformer_encoder_layers - i - 1
                                    filtered_dict = {k: v for k, v in filtered_dict.items() if f'transformer_encoder.encoder.layers.{removed_transformer_layer_index}' not in k}
                            model_updated_state_dict = Timeseries_model.state_dict()
                            model_updated_state_dict.update(filtered_dict)
                            Timeseries_model.load_state_dict(model_updated_state_dict)
                        else:
                            f.write(f'Loading model {conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_encoder_name_finetuned}...\n')
                            f.write(f'Loading model {conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_timeseries_name_finetuned}...\n')
                            Encoder_model.load_state_dict(load(f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_encoder_name_finetuned}',
                                                               map_location=device))
                            Timeseries_model.load_state_dict(load(f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_timeseries_name_finetuned}',
                                                               map_location=device))
                    
                    best_model_encoder = deepcopy(Encoder_model.state_dict())
                    best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Done!\n\n')
                except (FileNotFoundError, RuntimeError):
                    best_model_encoder = None
                    best_model_timeseries = None
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('An error occurred while loading the files! Training without loaded model weights...\n\n')
                
                if finetuning_stage == 1:
                    # We only fine-tune the classification layers in the first fine-tuning stage
                    for name, param in Encoder_model.named_parameters():
                        param.requires_grad = False
                    for name, param in Timeseries_model.named_parameters():
                        if not 'classification' in name:
                            param.requires_grad = False
                
                # Flag for indicating if max epochs are reached
                max_epochs_reached = 1
                
                # Start training our model
                if conf.train_model:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Starting training...\n')
                    
                    for epoch in range(1, conf.max_epochs + 1):
                        
                        start_time = time.time()
                
                        # Lists containing the losses of each epoch
                        epoch_loss_training = []
                        epoch_loss_validation = []
                        epoch_true_Y_training = np.array([])
                        epoch_pred_Y_training = np.array([])
                        epoch_true_Y_validation = np.array([])
                        epoch_pred_Y_validation = np.array([])
                
                        # Indicate that we are in training mode, so e.g. dropout will function
                        Encoder_model.train()
                        Timeseries_model.train()
                        
                        # Loop through every minibatch of our training data
                        for train_data in train_data_loader:
                            
                            # Get the minibatches
                            X, Y, data_mask, _ = [element.to(device) for element in train_data] # yksi _ liikaa
                            
                            # Zero the gradient of the optimizer
                            optimizer.zero_grad()
                            
                            X_input = []
                            data_masks = []
                            Y_orig = []
                            
                            for sequence_index in range(len(X)):
                                if not data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                                    X_input.append(X[sequence_index])
                                    Y_orig.append(Y[sequence_index])
                                    data_masks.append(data_mask[sequence_index])
                            if len(X_input) == 0:
                                continue
                            
                            X_input = stack(X_input, dim=0)
                            Y_orig = stack(Y_orig, dim=0)
                            data_masks = stack(data_masks, dim=0)
                            padding_masks = data_masks.bool()
                            
                            # Pass our data through the encoder and the time-series model
                            sensor_encoding = Encoder_model(X_input.float())
                            Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_masks)
                            
                            # We only take the unmasked frames into account
                            Y_pred_unmasked = Y_pred[~padding_masks, :]
                            Y_unmasked = Y_orig[~padding_masks, :].max(dim=1)[1]
                            
                            # Compute the weighted loss
                            loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                            
                            # Compute the prediction accuracy of the model (unweighted mean F1 score)
                            smax = Softmax(dim=1)
                            Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                            predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                            epoch_true_Y_training = np.concatenate((epoch_true_Y_training, Y_unmasked.detach().cpu().numpy()))
                            epoch_pred_Y_training = np.concatenate((epoch_pred_Y_training, predictions))
                            
                            # Perform the backward pass
                            loss.backward()
                            
                            # Update the weights
                            optimizer.step()
                            
                            # Add the loss to the total loss of the epoch
                            epoch_loss_training.append(loss.item())
                        
                        
                        # Indicate that we are in evaluation mode, so e.g. dropout will not function
                        Encoder_model.eval()
                        Timeseries_model.eval()
                
                        # Make PyTorch not calculate the gradients, so everything will be much faster.
                        with no_grad():
                            
                            # Loop through every batch of our validation data and perform a similar process
                            # as for the training data
                            for validation_data in validation_data_loader:
                                X, Y, data_mask, _= [element.to(device) for element in validation_data]
                                X_input = []
                                data_masks = []
                                Y_orig = []
                                for sequence_index in range(len(X)):
                                    if not data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                                        X_input.append(X[sequence_index])
                                        Y_orig.append(Y[sequence_index])
                                        data_masks.append(data_mask[sequence_index])
                                if len(X_input) == 0:
                                    continue
                                X_input = stack(X_input, dim=0)
                                Y_orig = stack(Y_orig, dim=0)
                                data_masks = stack(data_masks, dim=0)
                                padding_masks = data_masks.bool()
                                sensor_encoding = Encoder_model(X_input.float())
                                Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_masks)
                                Y_pred_unmasked = Y_pred[~padding_masks, :]
                                Y_unmasked = Y_orig[~padding_masks, :].max(dim=1)[1]
                                loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                                smax = Softmax(dim=1)
                                Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                                predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                                epoch_true_Y_validation = np.concatenate((epoch_true_Y_validation, Y_unmasked.detach().cpu().numpy()))
                                epoch_pred_Y_validation = np.concatenate((epoch_pred_Y_validation, predictions))
                                epoch_loss_validation.append(loss.item())
                        
                        # Calculate mean losses
                        epoch_loss_training = np.array(epoch_loss_training).mean()
                        epoch_loss_validation = np.array(epoch_loss_validation).mean()
                        if conf.train_criterion == 'f1':
                            epoch_accuracy_training = f1_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                            epoch_accuracy_validation = f1_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                        elif conf.train_criterion == 'recall':
                            epoch_accuracy_training = recall_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                            epoch_accuracy_validation = recall_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                        else:
                            sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                        
                        # Check early stopping conditions
                        if epoch_accuracy_validation > highest_validation_accuracy:
                            highest_validation_accuracy = epoch_accuracy_validation
                            patience_counter = 0
                            best_model_encoder = deepcopy(Encoder_model.state_dict())
                            best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                            best_validation_epoch = epoch
                            
                            # We first make sure that we are able to write the files
                            save_names = [f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_encoder_name_finetuned}',
                                          f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_timeseries_name_finetuned}']
                            for model_save_name in save_names:
                                model_path, model_filename = os.path.split(model_save_name)
                                if not os.path.exists(model_path):
                                    if model_path != '':
                                        os.makedirs(model_path)
                            
                            save(best_model_encoder, f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_encoder_name_finetuned}')
                            save(best_model_timeseries, f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_timeseries_name_finetuned}')
                        else:
                            patience_counter += 1
                        
                        end_time = time.time()
                        epoch_time = end_time - start_time
                        
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                              f'Mean validation loss: {epoch_loss_validation:6.4f} | '
                              f'Mean training accuracy: {epoch_accuracy_training:6.4f} | '
                              f'Mean validation accuracy: {epoch_accuracy_validation:6.4f} (highest: {highest_validation_accuracy:6.4f}) | '
                              f'Duration: {epoch_time:4.2f} seconds\n')
                        
                        # We check that do we need to update the learning rate based on the validation loss
                        if conf.use_lr_scheduler:
                            if finetuning_stage == 1:
                                if conf.lr_scheduler_stage_1 == 'ReduceLROnPlateau':
                                    lr_scheduler.step(epoch_accuracy_validation)
                                else:
                                    lr_scheduler.step()
                            else:
                                if epoch < conf.lr_scheduler_stage_2_part_1_epochs + 1:
                                    lr_scheduler = scheduler_part_1
                                    if conf.lr_scheduler_stage_2_part_1 == 'ReduceLROnPlateau':
                                        lr_scheduler.step(epoch_accuracy_validation)
                                    else:
                                        lr_scheduler.step()
                                else:
                                    lr_scheduler = scheduler_part_2
                                    if conf.lr_scheduler_stage_2_part_2 == 'ReduceLROnPlateau':
                                        lr_scheduler.step(epoch_accuracy_validation)
                                    else:
                                        lr_scheduler.step()
                            current_learning_rate = optimizer.param_groups[0]['lr']
                            if current_learning_rate != learning_rate:
                                learning_rate = current_learning_rate
                                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                                    f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
                        
                        # If patience counter is fulfilled, stop the training
                        if patience_counter >= conf.patience:
                            max_epochs_reached = 0
                            break
                    
                    if max_epochs_reached:
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write('\nMax number of epochs reached, stopping training\n\n')
                    else:
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write('\nExiting due to early stopping\n\n')
                    
                    if best_model_encoder is None:
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
                        sys.exit('No best model, exiting...')
                    else:
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write(f'\nBest epoch {best_validation_epoch} with validation accuracy {highest_validation_accuracy}\n\n')
            
            
            
            if conf.test_model:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nStarting testing... => ')
                    
                # Load the best version of the model
                try:
                    Encoder_model.load_state_dict(load(f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_encoder_name_finetuned}',
                                                       map_location=device))
                    Timeseries_model.load_state_dict(load(f'{conf.result_dir}/experiment_{conf.finetuning_experiment_number}/fold_{fold+1}/{best_model_timeseries_name_finetuned}',
                                                          map_location=device))
                except (FileNotFoundError, RuntimeError):
                    Encoder_model.load_state_dict(best_model_encoder)
                    Timeseries_model.load_state_dict(best_model_timeseries)
                        
                testing_loss = []
                epoch_true_Y_testing = np.array([])
                epoch_pred_Y_testing = np.array([])
                Encoder_model.eval()
                Timeseries_model.eval()
                with no_grad():
                    for test_data in test_data_loader:
                        X, Y, data_mask, _= [element.to(device) for element in test_data]
                        X_input = []
                        data_masks = []
                        Y_orig = []
                        for sequence_index in range(len(X)):
                            if not data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                                X_input.append(X[sequence_index])
                                Y_orig.append(Y[sequence_index])
                                data_masks.append(data_mask[sequence_index])
                        if len(X_input) == 0:
                            continue
                        X_input = stack(X_input, dim=0)
                        Y_orig = stack(Y_orig, dim=0)
                        data_masks = stack(data_masks, dim=0)
                        padding_masks = data_masks.bool()
                        sensor_encoding = Encoder_model(X_input.float())
                        Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_masks)
                        Y_pred_unmasked = Y_pred[~padding_masks, :]
                        Y_unmasked = Y_orig[~padding_masks, :].max(dim=1)[1]
                        loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                        smax = Softmax(dim=1)
                        Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                        predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                        epoch_true_Y_testing = np.concatenate((epoch_true_Y_testing, Y_unmasked.detach().cpu().numpy()))
                        epoch_pred_Y_testing = np.concatenate((epoch_pred_Y_testing, predictions))
                        testing_loss.append(loss.item())
                    testing_loss = np.array(testing_loss).mean()
                    
                    conf_mat = confusion_matrix(epoch_true_Y_testing, epoch_pred_Y_testing, labels=np.arange(7))
                    
                    if conf.train_criterion == 'f1':
                        testing_accuracy = f1_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
                    elif conf.train_criterion == 'recall':
                        testing_accuracy = recall_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
                    else:
                        sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                    
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write(f'Testing loss: {testing_loss:7.4f}, testing accuracy: {testing_accuracy:7.4f}\n\n\n\n\n\n')
                fold_test_accuracies.append(testing_accuracy)
                fold_conf_mats.append(conf_mat)
        
        fold_test_accuracies_mean = np.array(fold_test_accuracies).mean()
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('\n########################################################################################\n')
            f.write(f'Mean test accuracies of {conf.num_folds}-folds cross-validation: {fold_test_accuracies_mean}')
            f.write('\n########################################################################################\n\n\n\n')
        
        combined_conf_mat = sum(fold_conf_mats)
        np.save(f'{conf.result_dir}/confusion_matrix_experiment_{conf.finetuning_experiment_number}.npy', combined_conf_mat)
        output_categories = 7
        
        prec = np.zeros(output_categories)
        rec = np.zeros(output_categories)
        f1 = np.zeros(output_categories)
        acc = np.float32(np.sum(combined_conf_mat.diagonal()))/np.float32(np.sum(combined_conf_mat))
        for i in range(combined_conf_mat.shape[0]):
            # Check if target contains current category
            containsCat = np.sum(combined_conf_mat[:,i]) > 0
            if containsCat:
                prec[i] = np.float32(combined_conf_mat[i,i])/np.float32(np.sum(combined_conf_mat[i,:]))
                rec[i] = np.float32(combined_conf_mat[i,i])/np.float32(np.sum(combined_conf_mat[:,i]))
                if np.isnan(prec[i]):
                    prec[i] = 0.0
                if np.isnan(rec[i]):
                    rec[i] = 0.0    
                f1[i] = 2.0*prec[i]*rec[i]/(prec[i] + rec[i])
       
                if np.isnan(f1[i]):
                    f1[i] = 0.0        
            else:
                prec[i] = np.nan; rec[i] = np.nan; f1[i] = np.nan

        prec_mean = np.nanmean(prec)
        rec_mean = np.nanmean(rec) 
        f1_mean = np.nanmean(f1)
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('\n########################################################################################\n')
            f.write(f'Mean precision for {conf.num_folds}-folds cross-validation: {prec_mean}\n')
            f.write(f'Mean recall for {conf.num_folds}-folds cross-validation: {rec_mean}\n')
            f.write(f'Mean F1 score for {conf.num_folds}-folds cross-validation: {f1_mean}')
            f.write('\n########################################################################################\n\n\n\n')
