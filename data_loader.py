import numpy as np
from torch.utils.data import Dataset
import os
import sys
import librosa
import scipy


class imu_dataset(Dataset):

    def __init__(self, data_list, train_val_test = 'train', train_sequence_length = 260, train_val_ratio = 0.8,
                 random_seed = 42, window_len = 120, hop_len = 60, mix_train_val_babies = False,
                 augment_train_data = False, aug_p_noise = 0.0, aug_p_dropout = 0.1, aug_p_rotation = 0.3,
                 aug_p_chandropout = 0.3, aug_p_time_warping = 0.0, data_sampling_rate=1.0,
                 include_artificial_labels=False, normalize_functionals_sample_level=False,
                 normalize_functionals_dataset_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True):
        super().__init__()
        
        if train_val_test == 'train' and augment_train_data:
            self.augment = augment_train_data
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
        data_masks = []
        if not mix_train_val_babies and train_val_test != 'test':
            # We split our training and validation data so that baby-specific data is not included in both sets.
            num_train_babies = int(np.round(train_val_ratio*len(data_list)))
            train_val_babies_permutation = np.random.RandomState(seed=random_seed*2).permutation(len(data_list))
            if train_val_test == 'train':
                data_list = [data_list[i] for i in train_val_babies_permutation[:num_train_babies]]
            else:
                data_list = [data_list[i] for i in train_val_babies_permutation[num_train_babies:]]
        
        if include_artificial_labels:
            Y = []
        # We go through the data sequences one at a time and we append them to their appropriate lists.
        for baby_data in data_list:
            data_in = baby_data['X']
            data_mask = baby_data['Mask']
            if include_artificial_labels:    
                labels_in = baby_data['A1']  # handle Y here too
            
            num_sequences = data_in.shape[0] // train_sequence_length
            leftover_sequence_len = data_in.shape[0] % train_sequence_length
            
            if not mix_train_val_babies or train_val_test == 'test':
                for i in range(num_sequences):
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                    if include_artificial_labels:
                        Y.append(labels_in[i*train_sequence_length:(i+1)*train_sequence_length,:])
            else:
                num_train_seq = int(np.round(train_val_ratio*num_sequences))
                train_val_permutation = np.random.RandomState(seed=random_seed).permutation(num_sequences)
                sequences = train_val_permutation[:num_train_seq] if train_val_test == 'train' else train_val_permutation[num_train_seq:]
                
                for i in sequences:
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                    if include_artificial_labels:
                        Y.append(labels_in[i*train_sequence_length:(i+1)*train_sequence_length,:])
            
            if leftover_sequence_len != 0 and (train_val_test != 'validation' or not mix_train_val_babies):
                if not mix_train_val_babies or train_val_test == 'test':
                    last_i = num_sequences - 1
                else:
                    if len(sequences) == 0:
                        continue
                    last_i = sequences[-1]
                
                X_leftover = np.copy(data_in[last_i*train_sequence_length:(last_i+1)*train_sequence_length,:,:])
                X_leftover[:leftover_sequence_len] = data_in[-leftover_sequence_len:, :, :]
                X.append(X_leftover)
                leftover_mask = np.ones_like(data_mask[last_i*train_sequence_length:(last_i+1)*train_sequence_length])
                leftover_mask[:leftover_sequence_len] = data_mask[-leftover_sequence_len:]
                data_masks.append(leftover_mask)
                
                if include_artificial_labels:
                    Y_left = np.copy(labels_in[last_i*train_sequence_length:(last_i+1)*train_sequence_length,:])
                    Y_left[:leftover_sequence_len] = labels_in[-leftover_sequence_len:, :]
                    Y.append(Y_left)
        print(f"Total sequences in X: {len(X)}")
        
        
        if include_artificial_labels:
            self.Y = np.array(Y)
        self.data_masks = np.array(data_masks)
        self.X = np.array(X)
                

        # We compute functionals of the features
        feats_functionals = []
        for feat in X:
            functionals = []
            if functionals_include_mean:
                functionals.append(np.mean(feat, axis=2))
            if functionals_include_var:
                functionals.append(np.var(feat, axis=2))
            if functionals_include_skew:
                functionals.append(scipy.stats.skew(feat, axis=2))
            if functionals_include_kurtosis:
                functionals.append(scipy.stats.kurtosis(feat, axis=2))
            if functionals_include_min:
                functionals.append(feat.min(axis=2))
            if functionals_include_max:
                functionals.append(feat.max(axis=2))
            if functionals_include_zcr:
                functionals.append(librosa.zero_crossings(feat, axis=2).sum(axis=2) / window_len)
            if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                if functionals_include_acf_mean:
                    ac_channel_mean = []
                if functionals_include_acf_var:
                    ac_channel_var = []
                if functionals_include_acf_skew:
                    ac_channel_skew = []
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = []
                for i in range(feat.shape[1]):
                    feat_channel = feat[:, i, :]
                    ac = estimated_autocorrelation(feat_channel)
                    if functionals_include_acf_mean:
                        ac_channel_mean.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        ac_channel_var.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        ac_channel_skew.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        ac_channel_kurtosis.append(scipy.stats.kurtosis(ac, axis=1))
                if functionals_include_acf_mean:
                    ac_channel_mean = np.transpose(np.array(ac_channel_mean))
                    functionals.append(ac_channel_mean)
                if functionals_include_acf_var:
                    ac_channel_var = np.transpose(np.array(ac_channel_var))
                    functionals.append(ac_channel_var)
                if functionals_include_acf_skew:
                    ac_channel_skew = np.transpose(np.array(ac_channel_skew))
                    functionals.append(ac_channel_skew)
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = np.transpose(np.array(ac_channel_kurtosis))
                    functionals.append(ac_channel_kurtosis)
            functionals = np.stack(functionals, axis=2)
            
            # We reshape the functional array from the shape [train_sequence_length, num_channels, num_functionals]
            # into the shape [train_sequence_length, num_channels * num_functionals]
            functionals = functionals.reshape(functionals.shape[0], -1)
            
            if normalize_functionals_sample_level:
                feats_functionals.append(normalize_sample(functionals))
            else:
                feats_functionals.append(functionals)
        
        if normalize_functionals_dataset_level:
            feats_functionals = normalize_dataset(feats_functionals)
        
        self.feats_functionals = np.array(feats_functionals)
        
        # We create artificial labels for our randomly generated dataset. There are nine different labels
        # for movement in MAIJU data.
        # Tämä muutetaan niin, että tuodaan oikeat labelit data_listista
        """         
        if include_artificial_labels:
            Y = np.zeros((len(self.X), train_sequence_length, 9))
            for i in range(len(Y)):
                for j in range(train_sequence_length):
                    random_vec = np.random.rand(Y.shape[2])
                    max_ind = np.argmax(random_vec)
                    Y[i, j, max_ind] = 1.0
        
            self.Y = Y
         """

        self.include_artificial_labels = include_artificial_labels
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :, :]
            self.data_masks = self.data_masks[sampling_indices, :]
            self.feats_functionals = self.feats_functionals[sampling_indices, :, :]
            if include_artificial_labels:
                self.Y = self.Y[sampling_indices, :, :]


    def __len__(self) -> int:
        return len(self.X)


    def __getitem__(self, index):
        """         
        # Global flag to avoid printing thousands of times
        if not hasattr(self, "_debug_printed"):
            print("DEBUG: __getitem__ called for index:", index)
            self._debug_printed = True 
        """


        if self.augment:
            X = data_augmentation(self.X[index], self.aug_p_noise, self.aug_p_dropout, self.aug_p_rotation,
                                  self.aug_p_chandropout, self.aug_p_time_warping, self.window_len, self.hop_len)
        else:
            X = self.X[index]
        
        if self.include_artificial_labels:
            target_labels = self.Y[index]
        else:
            target_labels = np.zeros((self.X.shape[1], 1), dtype=np.float32) # Previously just 0
        
        """ 
        # ?
        X, target_labels, mask, feats = (
            X,
            target_labels,
            self.data_masks[index],
            self.feats_functionals[index],
        )
        # Print shapes and dtypes once
        if not hasattr(self, "_debug_shapes_printed"):
            print("DEBUG SHAPES:")
            print("  X:", type(X), getattr(X, "shape", None))
            print("  target_labels:", type(target_labels), getattr(target_labels, "shape", None))
            print("  mask:", type(mask), getattr(mask, "shape", None))
            print("  feats:", type(feats), getattr(feats, "shape", None), "dtype:", getattr(feats, "dtype", None))
            self._debug_shapes_printed = True 
        """

        return X, target_labels, self.data_masks[index], self.feats_functionals[index]




# Normalize the 2D input sample to have zero mean and unit variance along each feature. The dimensions of
# the input are (frame_index, feature_index).
def normalize_sample(feats) -> np.ndarray:
    
    normalized = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    
    # Remove NaN values by converting them to zero
    normalized = np.nan_to_num(normalized)
    
    return normalized


# Normalize the 3D input features (can be different-length) to have zero mean and unit variance
# -> the input is a list with samples of dimensions (frame_index, feature_index).
def normalize_dataset(feat_list):
    
    feats_unrolled = np.nan_to_num(np.concatenate(feat_list, axis=0))
    feat_mean = feats_unrolled.mean(axis=0)
    feat_std = feats_unrolled.std(axis=0)
    del feats_unrolled
    
    for i in range(len(feat_list)):
        feat_list[i] = (feat_list[i] - feat_mean) / feat_std
        feat_list[i] = np.nan_to_num(feat_list[i]) # Remove NaN values by converting them to zero
    
    return feat_list


def estimated_autocorrelation(frames):
    
    ac = []
    for x in frames:
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode = 'full')[-n:]
        eps = 1e-8 # Add to avoid division by zero
        result = r/((variance + eps) * (np.arange(n, 0, -1)))
        ac.append(result)
        
    return np.array(ac)


def time_warping(data, p=1.0, winlen=120):
    basevec = np.arange(winlen) + 1.0
    Nframes = int(np.floor(((data.shape[0] - winlen)/winlen) + 1))
    for iFrame in range(Nframes):
        # Randomly warp p*100% of frames
        if np.random.random_sample() <= p:
        # Random sinusoid with random phase, amplitude [0.5, 1.5], frequency
            freq = np.random.random_sample() * basevec / basevec.shape[0]
            phase = 2 * np.pi * np.random.random_sample()
            amplitude = np.random.random_sample()
            sinusoid = amplitude * np.sin(2 * np.pi * freq + phase) + 2
            sinusoid /= np.mean(sinusoid)

            newbase = np.cumsum(sinusoid)
            start = iFrame * winlen
            stop = start + winlen
            for iChan in range(data.shape[1]):
                data[start:stop,iChan] = np.interp(newbase, basevec, data[start:stop, iChan])

    return data


def rotationMatrix(a_x, a_y, a_z, angle_type='deg'):
    if angle_type == 'deg':
        a_x *= np.pi / 180.0
        a_y *= np.pi / 180.0
        a_z *= np.pi / 180.0

    M = np.array([[np.cos(a_y) * np.cos(a_z), 
                   -np.cos(a_x) * np.sin(a_z) + np.sin(a_x) * np.sin(a_y) * np.cos(a_z), 
                   np.sin(a_x) * np.sin(a_z) + np.cos(a_x) * np.sin(a_y) * np.cos(a_z)],
                  [np.cos(a_y) * np.sin(a_z), 
                   np.cos(a_x) * np.cos(a_z) + np.sin(a_x) * np.sin(a_y) * np.sin(a_z),
                   -np.sin(a_x) * np.cos(a_z) + np.cos(a_x) * np.sin(a_y) * np.sin(a_z)],
                  [-np.sin(a_y),
                   np.sin(a_x) * np.cos(a_y),
                   np.cos(a_x) * np.cos(a_y)]])

    return M


def random_rotation(data, angle=15.0):
    # Get rotation matrix, random rotation for each sensor
    range_x = [-angle, angle]
    range_y = [-angle, angle]
    range_z = [-angle, angle]
    Nsens = data.shape[1] // 6
    n = data.shape[-1] // 2
    acc = data[:,:n]
    gyro = data[:,n:]
    for i in range(Nsens):
        a_x = np.random.random_sample() * (range_x[1] - range_x[0]) + range_x[0]
        a_y = np.random.random_sample() * (range_y[1] - range_y[0]) + range_y[0]
        a_z = np.random.random_sample() * (range_z[1] - range_z[0]) + range_z[0]
        M = rotationMatrix(a_x, a_y, a_z)
        
        acc[:,i*3:(i+1)*3] = np.matmul(acc[:,i*3:(i+1)*3], M)
        gyro[:,i*3:(i+1)*3] = np.matmul(gyro[:,i*3:(i+1)*3], M)

    data = np.concatenate([acc, gyro], axis=-1)

    return data


def dropout_noise(data, p):
    mask = np.random.binomial(1, 1.0 - p, data.shape)
    
    return data * mask


def channel_dropout(data, num_chans=1, tot_chans=3): # tot chans was 4
    chans_to_drop = np.random.permutation(tot_chans)
    chans_to_drop = chans_to_drop[:num_chans]
    N = data.shape[-1] // 2
    for i in chans_to_drop:
        data[:,(3*i):(3*i+3)] *= 0.0 # Accelerometer signals
        data[:,(N+3*i):(N+3*i+3)] *= 0.0 # Gyroscope signals

    return data


def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed


def data_augmentation(data, aug_p_noise, aug_p_dropout, aug_p_rotation, aug_p_chandropout,
                      aug_p_time_warping, window_len, hop_len):
    
    # Augmentation to frames, assume data is 50% overlapped
    N = data.shape[-1] // 2
    data = np.concatenate([np.reshape(np.transpose(data[:,:,:N], [0,2,1]), [-1, data.shape[1]]),
                           np.transpose(data[-1,:,N:])], axis=0)

    # Time warping
    if np.random.random_sample() < aug_p_time_warping:
        data = time_warping(data, p=1.0, winlen=window_len)

    # Random rotation
    if np.random.random_sample() < aug_p_rotation:
        data = random_rotation(data)

    # Additive noise augmentation
    if np.random.random_sample() < aug_p_noise:
        data = dropout_noise(data, aug_p_dropout)

    # Sensor dropout
    if np.random.random_sample() < aug_p_chandropout:
        data = channel_dropout(data, num_chans=1)

    # Retain framed format
    data = frame_sig(data, window_len, hop_len)

    return data


def frame_sig_eeg(X, winlen, hop):
    """
    The input data should be either of size [num_sequences, num_channels, sequence_length]
    or of size [num_sequences, sequence_length]
    
    Output is of size [num_sequences, Nframes, num_channels, winlen]
    """
    
    if len(X.shape) < 3:
        # We add a dummy channel to the data
        X = np.expand_dims(X, axis=1)
    
    Nframes = int(np.floor(((X.shape[2] - winlen)/hop) + 1))
    num_channels = X.shape[1]
    num_sequences = X.shape[0]
    
    X_framed = np.zeros([num_sequences, Nframes, num_channels, winlen], dtype=np.float32)
    for i in range(num_sequences):
        for j in range(0, Nframes):
            start = j * hop
            stop = start + winlen
            X_framed[i,j,:,:] = X[i,:,start:stop]

    return X_framed
