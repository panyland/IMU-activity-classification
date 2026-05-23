import numpy as np
from torch.utils.data import Dataset


class imu_dataset(Dataset):

    def __init__(self, data_list, train_val_test='train', train_sequence_length=260, train_val_ratio=0.8,
                 random_seed=42, window_len=120, hop_len=60, mix_train_val_subjects=False,
                 augment_train_data=False, aug_p_noise=0.0, aug_p_dropout=0.1, aug_p_rotation=0.3,
                 aug_p_chandropout=0.3, aug_p_time_warping=0.0, data_sampling_rate=1.0,
                 label_key='A1'):
        super().__init__()

        if train_val_test == 'train' and augment_train_data:
            self.augment = True
            self.aug_p_noise = aug_p_noise
            self.aug_p_dropout = aug_p_dropout
            self.aug_p_rotation = aug_p_rotation
            self.aug_p_chandropout = aug_p_chandropout
            self.aug_p_time_warping = aug_p_time_warping
            self.window_len = window_len
            self.hop_len = hop_len
        else:
            self.augment = False

        X = []
        Y = []
        data_masks = []

        if not mix_train_val_subjects and train_val_test != 'test':
            num_train = int(np.round(train_val_ratio * len(data_list)))
            perm = np.random.RandomState(seed=random_seed * 2).permutation(len(data_list))
            if train_val_test == 'train':
                data_list = [data_list[i] for i in perm[:num_train]]
            else:
                data_list = [data_list[i] for i in perm[num_train:]]

        for subject_data in data_list:
            data_in = subject_data['X']
            data_mask = subject_data['Mask']
            labels_in = subject_data[label_key]

            num_sequences = data_in.shape[0] // train_sequence_length
            leftover_len = data_in.shape[0] % train_sequence_length

            if not mix_train_val_subjects or train_val_test == 'test':
                for i in range(num_sequences):
                    s = i * train_sequence_length
                    e = s + train_sequence_length
                    X.append(data_in[s:e])
                    Y.append(labels_in[s:e])
                    data_masks.append(data_mask[s:e])
            else:
                num_train_seq = int(np.round(train_val_ratio * num_sequences))
                perm = np.random.RandomState(seed=random_seed).permutation(num_sequences)
                sequences = perm[:num_train_seq] if train_val_test == 'train' else perm[num_train_seq:]
                for i in sequences:
                    s = i * train_sequence_length
                    e = s + train_sequence_length
                    X.append(data_in[s:e])
                    Y.append(labels_in[s:e])
                    data_masks.append(data_mask[s:e])

            if leftover_len != 0 and (train_val_test != 'validation' or not mix_train_val_subjects):
                if not mix_train_val_subjects or train_val_test == 'test':
                    last_i = num_sequences - 1
                else:
                    if len(sequences) == 0:
                        continue
                    last_i = sequences[-1]

                X_left = np.copy(data_in[last_i * train_sequence_length:(last_i + 1) * train_sequence_length])
                X_left[:leftover_len] = data_in[-leftover_len:]
                X.append(X_left)

                Y_left = np.copy(labels_in[last_i * train_sequence_length:(last_i + 1) * train_sequence_length])
                Y_left[:leftover_len] = labels_in[-leftover_len:]
                Y.append(Y_left)

                mask_left = np.ones_like(data_mask[last_i * train_sequence_length:(last_i + 1) * train_sequence_length])
                mask_left[:leftover_len] = data_mask[-leftover_len:]
                data_masks.append(mask_left)

        print(f"Total sequences ({train_val_test}): {len(X)}")

        self.X = np.array(X)
        self.Y = np.array(Y)
        self.data_masks = np.array(data_masks)

        if data_sampling_rate < 1.0 and train_val_test != 'test':
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3 * random_seed)
            idx = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[idx]
            self.Y = self.Y[idx]
            self.data_masks = self.data_masks[idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.augment:
            X = data_augmentation(self.X[index], self.aug_p_noise, self.aug_p_dropout,
                                  self.aug_p_rotation, self.aug_p_chandropout,
                                  self.aug_p_time_warping, self.window_len, self.hop_len)
        else:
            X = self.X[index]

        return X, self.Y[index], self.data_masks[index]


def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen) / hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32)
    for i in range(Nframes):
        start = i * hop
        X_framed[i, :, :] = np.transpose(X[start:start + winlen, :])
    return X_framed


def time_warping(data, p=1.0, winlen=120):
    basevec = np.arange(winlen) + 1.0
    Nframes = int(np.floor(((data.shape[0] - winlen) / winlen) + 1))
    for iFrame in range(Nframes):
        if np.random.random_sample() <= p:
            freq = np.random.random_sample() * basevec / basevec.shape[0]
            phase = 2 * np.pi * np.random.random_sample()
            amplitude = np.random.random_sample()
            sinusoid = amplitude * np.sin(2 * np.pi * freq + phase) + 2
            sinusoid /= np.mean(sinusoid)
            newbase = np.cumsum(sinusoid)
            start = iFrame * winlen
            for iChan in range(data.shape[1]):
                data[start:start + winlen, iChan] = np.interp(newbase, basevec, data[start:start + winlen, iChan])
    return data


def rotationMatrix(a_x, a_y, a_z, angle_type='deg'):
    if angle_type == 'deg':
        a_x *= np.pi / 180.0
        a_y *= np.pi / 180.0
        a_z *= np.pi / 180.0
    M = np.array([
        [np.cos(a_y) * np.cos(a_z),
         -np.cos(a_x) * np.sin(a_z) + np.sin(a_x) * np.sin(a_y) * np.cos(a_z),
         np.sin(a_x) * np.sin(a_z) + np.cos(a_x) * np.sin(a_y) * np.cos(a_z)],
        [np.cos(a_y) * np.sin(a_z),
         np.cos(a_x) * np.cos(a_z) + np.sin(a_x) * np.sin(a_y) * np.sin(a_z),
         -np.sin(a_x) * np.cos(a_z) + np.cos(a_x) * np.sin(a_y) * np.sin(a_z)],
        [-np.sin(a_y),
         np.sin(a_x) * np.cos(a_y),
         np.cos(a_x) * np.cos(a_y)]
    ])
    return M


def random_rotation(data, angle=15.0):
    Nsens = data.shape[1] // 6
    n = data.shape[-1] // 2
    acc = data[:, :n]
    gyro = data[:, n:]
    for i in range(Nsens):
        a_x = np.random.random_sample() * 2 * angle - angle
        a_y = np.random.random_sample() * 2 * angle - angle
        a_z = np.random.random_sample() * 2 * angle - angle
        M = rotationMatrix(a_x, a_y, a_z)
        acc[:, i * 3:(i + 1) * 3] = np.matmul(acc[:, i * 3:(i + 1) * 3], M)
        gyro[:, i * 3:(i + 1) * 3] = np.matmul(gyro[:, i * 3:(i + 1) * 3], M)
    return np.concatenate([acc, gyro], axis=-1)


def dropout_noise(data, p):
    return data * np.random.binomial(1, 1.0 - p, data.shape)


def channel_dropout(data, num_chans=1, tot_chans=3):
    chans_to_drop = np.random.permutation(tot_chans)[:num_chans]
    N = data.shape[-1] // 2
    for i in chans_to_drop:
        data[:, 3 * i:3 * i + 3] *= 0.0
        data[:, N + 3 * i:N + 3 * i + 3] *= 0.0
    return data


def data_augmentation(data, aug_p_noise, aug_p_dropout, aug_p_rotation, aug_p_chandropout,
                      aug_p_time_warping, window_len, hop_len):
    N = data.shape[-1] // 2
    data = np.concatenate([
        np.reshape(np.transpose(data[:, :, :N], [0, 2, 1]), [-1, data.shape[1]]),
        np.transpose(data[-1, :, N:])
    ], axis=0)

    if np.random.random_sample() < aug_p_time_warping:
        data = time_warping(data, p=1.0, winlen=window_len)
    if np.random.random_sample() < aug_p_rotation:
        data = random_rotation(data)
    if np.random.random_sample() < aug_p_noise:
        data = dropout_noise(data, aug_p_dropout)
    if np.random.random_sample() < aug_p_chandropout:
        data = channel_dropout(data, num_chans=1)

    return frame_sig(data, window_len, hop_len)
