import numpy as np
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from scipy.io import loadmat
from py_conf_file_into_text import convert_py_conf_file_to_text
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from copy import deepcopy

import torch
from torch import cuda, no_grad, save, load, stack, from_numpy
from torch.utils.data import DataLoader

import conf_models as conf
from models import CNN_encoder, WaveNet
from data_loader import imu_dataset, frame_sig

imu_loss = getattr(__import__('torch.nn', fromlist=[conf.loss_name]), conf.loss_name)
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]),
                                 conf.optimization_algorithm)


def save_conf_matrix_plot(conf_mat, class_names, path, title):
    norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(len(class_names) + 2, len(class_names) + 1))
    sns.heatmap(norm, annot=conf_mat, fmt='d', cmap='Blues', vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Recall'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _collect_unmasked(X_batch, Y_batch, mask_batch):
    """Filter sequences with >95% padding and stack the rest."""
    X_input, Y_out, masks_out = [], [], []
    for i in range(len(X_batch)):
        if not mask_batch[i].sum() > 0.95 * X_batch.size(1):
            X_input.append(X_batch[i])
            Y_out.append(Y_batch[i])
            masks_out.append(mask_batch[i])
    if not X_input:
        return None, None, None
    return stack(X_input), stack(Y_out), stack(masks_out).bool()


if __name__ == "__main__":

    os.makedirs(conf.result_dir, exist_ok=True)
    log_path = f'{conf.result_dir}/{conf.name_of_log_textfile}'
    open(log_path, 'w').close()

    def log(msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    if conf.print_conf_contents:
        lines = convert_py_conf_file_to_text('conf_models.py')
        log('Configuration settings (conf_models.py):\n')
        for line in lines:
            log(line)
        log('\n' + '#' * 88 + '\n')

    device = 'cpu'  # GTX 1070 (sm_61) incompatible with installed PyTorch (sm_75+)
    log(f'Device: {device}\n')
    log('Loading data...')

    Data = []
    for mat_path in sorted(Path('./annotated_IMU_data/').glob('*.mat')):
        data = loadmat(mat_path)
        acc = data['acc_data']
        gyro = data['gyro_data']
        X_framed = frame_sig(np.concatenate((acc, gyro), axis=1), conf.window_len, conf.hop_len)
        Nframes = X_framed.shape[0]

        raw_labels = data[conf.label_key]
        labels_framed = np.zeros((Nframes, raw_labels.shape[1]), dtype=raw_labels.dtype)
        for i in range(Nframes):
            center = i * conf.hop_len + conf.window_len // 2
            labels_framed[i] = raw_labels[center]

        if Nframes < conf.train_sequence_length:
            pad = conf.train_sequence_length - Nframes
            X_framed = np.pad(X_framed, ((0, pad), (0, 0), (0, 0)))
            labels_framed = np.pad(labels_framed, ((0, pad), (0, 0)))
            mask = np.ones(conf.train_sequence_length)
            mask[:Nframes] = 0
        else:
            mask = np.zeros(Nframes)

        data['X'] = X_framed
        data[conf.label_key] = labels_framed
        data['Mask'] = mask
        Data.append(data)

    log(f'Loaded {len(Data)} subjects.\n')

    class_weights = from_numpy(1 / conf.prior_prob).to(device).float()

    N_subjects = len(Data)
    num_per_fold = N_subjects // conf.num_folds

    if conf.randomize_order_kfolds:
        perm = np.random.RandomState(seed=42).permutation(N_subjects)
    else:
        perm = np.arange(N_subjects)

    scorer = f1_score if conf.train_criterion == 'f1' else recall_score

    fold_test_scores = []
    fold_conf_mats = []

    for fold in range(conf.num_folds):
        log(f'{"#"*40}\nFold {fold+1}/{conf.num_folds}\n{"#"*40}')

        test_inds = set(perm[fold * num_per_fold:] if fold == conf.num_folds - 1
                        else perm[fold * num_per_fold:(fold + 1) * num_per_fold])
        D_train = [Data[i] for i in range(N_subjects) if i not in test_inds]
        D_test  = [Data[i] for i in range(N_subjects) if i in test_inds]

        Encoder_model = CNN_encoder(**conf.encoder_model_params).to(device)
        Timeseries_model = WaveNet(**conf.timeseries_model_params).to(device)

        all_params = list(Encoder_model.parameters()) + list(Timeseries_model.parameters())
        optimizer = optimization_algorithm(params=all_params, **conf.optimization_algorithm_params)
        current_lr = optimizer.param_groups[0]['lr']

        if conf.use_lr_scheduler:
            sched_cls = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_name]),
                                conf.lr_scheduler_name)
            lr_scheduler = sched_cls(optimizer, **conf.lr_scheduler_params)

        loss_function = (imu_loss(weight=class_weights, **conf.loss_params) if conf.use_class_weights
                         else imu_loss(**conf.loss_params))

        save_dir = f'{conf.result_dir}/fold_{fold+1}'
        os.makedirs(save_dir, exist_ok=True)
        encoder_path = f'{save_dir}/encoder_exp{conf.experiment_number}.pt'
        wavenet_path = f'{save_dir}/wavenet_exp{conf.experiment_number}.pt'

        best_encoder_state = None
        best_wavenet_state = None
        best_val_score = -1e10
        best_val_epoch = 0
        patience_counter = 0

        if conf.train_model:
            log('Initializing datasets...')
            train_set = imu_dataset(D_train, train_val_test='train', **conf.params_train_dataset)
            val_set   = imu_dataset(D_train, train_val_test='validation', **conf.params_validation_dataset)
            train_loader = DataLoader(train_set, **conf.params_train)
            val_loader   = DataLoader(val_set, **conf.params_train)
            log('Starting training...')

            stopped_early = False
            for epoch in range(1, conf.max_epochs + 1):
                t0 = time.time()

                Encoder_model.train()
                Timeseries_model.train()
                train_losses, true_train, pred_train = [], np.array([]), np.array([])

                for X_b, Y_b, mask_b in train_loader:
                    X_b, Y_b, mask_b = X_b.to(device), Y_b.to(device), mask_b.to(device)
                    X_in, Y_in, pmask = _collect_unmasked(X_b, Y_b, mask_b)
                    if X_in is None:
                        continue

                    optimizer.zero_grad()
                    encoding = Encoder_model(X_in.float())
                    Y_pred, _ = Timeseries_model(encoding)

                    Y_pred_um = Y_pred[~pmask]
                    Y_true_um = Y_in[~pmask].max(dim=1)[1]

                    loss = loss_function(Y_pred_um, Y_true_um)
                    loss.backward()
                    optimizer.step()

                    preds = np.argmax(torch.softmax(Y_pred_um, dim=1).detach().cpu().numpy(), axis=1)
                    true_train = np.concatenate((true_train, Y_true_um.detach().cpu().numpy()))
                    pred_train = np.concatenate((pred_train, preds))
                    train_losses.append(loss.item())

                Encoder_model.eval()
                Timeseries_model.eval()
                val_losses, true_val, pred_val = [], np.array([]), np.array([])

                with no_grad():
                    for X_b, Y_b, mask_b in val_loader:
                        X_b, Y_b, mask_b = X_b.to(device), Y_b.to(device), mask_b.to(device)
                        X_in, Y_in, pmask = _collect_unmasked(X_b, Y_b, mask_b)
                        if X_in is None:
                            continue

                        encoding = Encoder_model(X_in.float())
                        Y_pred, _ = Timeseries_model(encoding)

                        Y_pred_um = Y_pred[~pmask]
                        Y_true_um = Y_in[~pmask].max(dim=1)[1]

                        loss = loss_function(Y_pred_um, Y_true_um)
                        preds = np.argmax(torch.softmax(Y_pred_um, dim=1).detach().cpu().numpy(), axis=1)
                        true_val = np.concatenate((true_val, Y_true_um.detach().cpu().numpy()))
                        pred_val = np.concatenate((pred_val, preds))
                        val_losses.append(loss.item())

                train_score = scorer(true_train, pred_train, average='macro', zero_division=0)
                val_score   = scorer(true_val,   pred_val,   average='macro', zero_division=0)

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_epoch = epoch
                    patience_counter = 0
                    best_encoder_state = deepcopy(Encoder_model.state_dict())
                    best_wavenet_state = deepcopy(Timeseries_model.state_dict())
                    save(best_encoder_state, encoder_path)
                    save(best_wavenet_state, wavenet_path)
                else:
                    patience_counter += 1

                duration = time.time() - t0
                log(f'Epoch {epoch:04d} | '
                    f'train loss {np.mean(train_losses):.4f} | val loss {np.mean(val_losses):.4f} | '
                    f'train {conf.train_criterion} {train_score:.4f} | '
                    f'val {conf.train_criterion} {val_score:.4f} (best {best_val_score:.4f}) | '
                    f'{duration:.1f}s')

                if conf.use_lr_scheduler:
                    if conf.lr_scheduler_name == 'ReduceLROnPlateau':
                        lr_scheduler.step(val_score)
                    else:
                        lr_scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        current_lr = new_lr
                        log(f'LR updated to {current_lr}')

                if patience_counter >= conf.patience:
                    stopped_early = True
                    break

            log(f'\n{"Early stopping" if stopped_early else "Max epochs reached"}')
            log(f'Best epoch {best_val_epoch}, val {conf.train_criterion} = {best_val_score:.4f}\n')

        if conf.test_model:
            log('Initializing test set...')
            test_set = imu_dataset(D_test, train_val_test='test', **conf.params_test_dataset)
            test_loader = DataLoader(test_set, **conf.params_test)

            try:
                Encoder_model.load_state_dict(load(encoder_path, map_location=device))
                Timeseries_model.load_state_dict(load(wavenet_path, map_location=device))
            except (FileNotFoundError, RuntimeError):
                Encoder_model.load_state_dict(best_encoder_state)
                Timeseries_model.load_state_dict(best_wavenet_state)

            Encoder_model.eval()
            Timeseries_model.eval()
            test_losses, true_test, pred_test = [], np.array([]), np.array([])

            with no_grad():
                for X_b, Y_b, mask_b in test_loader:
                    X_b, Y_b, mask_b = X_b.to(device), Y_b.to(device), mask_b.to(device)
                    X_in, Y_in, pmask = _collect_unmasked(X_b, Y_b, mask_b)
                    if X_in is None:
                        continue

                    encoding = Encoder_model(X_in.float())
                    Y_pred, _ = Timeseries_model(encoding)

                    Y_pred_um = Y_pred[~pmask]
                    Y_true_um = Y_in[~pmask].max(dim=1)[1]

                    loss = loss_function(Y_pred_um, Y_true_um)
                    preds = np.argmax(torch.softmax(Y_pred_um, dim=1).detach().cpu().numpy(), axis=1)
                    true_test = np.concatenate((true_test, Y_true_um.detach().cpu().numpy()))
                    pred_test = np.concatenate((pred_test, preds))
                    test_losses.append(loss.item())

            conf_mat = confusion_matrix(true_test, pred_test, labels=np.arange(conf.num_classes))
            test_score = scorer(true_test, pred_test, average='macro', zero_division=0)

            log(f'Test loss: {np.mean(test_losses):.4f} | Test {conf.train_criterion}: {test_score:.4f}\n')
            fold_test_scores.append(test_score)
            fold_conf_mats.append(conf_mat)

            np.save(f'{save_dir}/conf_mat_fold{fold+1}.npy', conf_mat)
            save_conf_matrix_plot(conf_mat, conf.class_names,
                                  f'{save_dir}/conf_mat_fold{fold+1}.png',
                                  f'Fold {fold+1} — {conf.train_criterion.upper()} {test_score:.3f}')

    log('\n' + '#' * 88)
    log(f'Mean test {conf.train_criterion} over {conf.num_folds} folds: {np.mean(fold_test_scores):.4f}')
    log('#' * 88 + '\n')

    combined_conf_mat = sum(fold_conf_mats)
    np.save(f'{conf.result_dir}/conf_mat_combined.npy', combined_conf_mat)
    save_conf_matrix_plot(combined_conf_mat, conf.class_names,
                          f'{conf.result_dir}/conf_mat_combined.png',
                          f'LOSO Combined ({conf.num_folds} folds) — mean {conf.train_criterion.upper()} {np.mean(fold_test_scores):.3f}')

    prec = np.zeros(conf.num_classes)
    rec  = np.zeros(conf.num_classes)
    f1   = np.zeros(conf.num_classes)
    overall_acc = np.sum(combined_conf_mat.diagonal()) / np.sum(combined_conf_mat)

    for i in range(conf.num_classes):
        col_sum = np.sum(combined_conf_mat[:, i])
        row_sum = np.sum(combined_conf_mat[i, :])
        if col_sum > 0:
            prec[i] = combined_conf_mat[i, i] / row_sum if row_sum > 0 else 0.0
            rec[i]  = combined_conf_mat[i, i] / col_sum
            prec[i] = np.nan_to_num(prec[i])
            rec[i]  = np.nan_to_num(rec[i])
            f1[i]   = 2 * prec[i] * rec[i] / (prec[i] + rec[i]) if (prec[i] + rec[i]) > 0 else 0.0
        else:
            prec[i] = rec[i] = f1[i] = np.nan

    log('#' * 88)
    log(f'Overall accuracy : {overall_acc:.4f}')
    log(f'Mean precision   : {np.nanmean(prec):.4f}')
    log(f'Mean recall      : {np.nanmean(rec):.4f}')
    log(f'Mean F1          : {np.nanmean(f1):.4f}')
    log('#' * 88)

    # Per-class metrics CSV
    class_df = pd.DataFrame({
        'class': conf.class_names,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    })
    class_df.to_csv(f'{conf.result_dir}/per_class_metrics.csv', index=False, float_format='%.4f')

    # Per-fold scores CSV
    fold_df = pd.DataFrame({
        'fold': range(1, conf.num_folds + 1),
        conf.train_criterion: fold_test_scores,
    })
    fold_df.to_csv(f'{conf.result_dir}/per_fold_scores.csv', index=False, float_format='%.4f')
