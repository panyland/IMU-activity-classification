import numpy as np
import torch
from torch.nn import (Module, ModuleList, Linear, Conv1d, Conv2d, Conv3d,
                      BatchNorm1d, LayerNorm, Identity,
                      AvgPool2d, AvgPool3d,
                      Dropout, Tanh, Sigmoid, LeakyReLU, ReLU)
import sys


class CNN_encoder(Module):

    def __init__(self,
                 s_channels=18,
                 input_channels=120,
                 latent_channels=70,
                 output_channels=140,
                 dropout=0.3,
                 conv_1_kernel_size=(1, 3, 11),
                 conv_2_kernel_size=(1, 3, 5),
                 conv_3_kernel_size=(1, 3),
                 conv_4_kernel_size=(1, 4),
                 conv_1_stride=(1, 3, 6),
                 conv_2_stride=(1, 1, 2),
                 conv_3_stride=1,
                 conv_4_stride=1,
                 conv_1_zero_padding=(0, 0, 0),
                 conv_2_zero_padding=(0, 1, 2),
                 conv_3_zero_padding='same',
                 conv_4_zero_padding='valid',
                 pooling_1_zero_padding=(0, 0, 0),
                 pooling_1_kernel_size=(1, 1, 10),
                 pooling_2_zero_padding=(0, 0),
                 pooling_2_kernel_size=(1, 3),
                 normalization_type='layernorm'):

        super().__init__()

        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
        elif normalization_type is None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')

        self.r = np.int32(s_channels / 2)
        self.normalization_type = normalization_type

        self.conv_1_1 = Conv3d(1, latent_channels, conv_1_kernel_size, conv_1_stride, conv_1_zero_padding)
        self.conv_1_2 = Conv3d(1, latent_channels, conv_1_kernel_size, conv_1_stride, conv_1_zero_padding)
        self.conv_1_3 = Conv3d(1, latent_channels, conv_1_kernel_size, conv_1_stride, conv_1_zero_padding)

        self.conv_2_1 = Conv3d(latent_channels, latent_channels, conv_2_kernel_size, conv_2_stride, conv_2_zero_padding)
        self.conv_2_2 = Conv3d(latent_channels, latent_channels, conv_2_kernel_size, conv_2_stride, conv_2_zero_padding)
        self.conv_2_3 = Conv3d(latent_channels, latent_channels, conv_2_kernel_size, conv_2_stride, conv_2_zero_padding)

        self.conv_3 = Conv2d(latent_channels, output_channels, conv_3_kernel_size, conv_3_stride, conv_3_zero_padding)
        self.conv_4 = Conv2d(output_channels, output_channels, conv_4_kernel_size, conv_4_stride, conv_4_zero_padding)

        self.pooling_1 = AvgPool3d(pooling_1_kernel_size, padding=pooling_1_zero_padding)
        self.pooling_2 = AvgPool2d(pooling_2_kernel_size, padding=pooling_2_zero_padding)

        self.normalization = normalization_layer(output_channels)
        self.tanh = Tanh()
        self.lrelu = LeakyReLU()
        self.dropout = Dropout(dropout)

    def _conv_module(self, X, conv1, conv2):
        X = self.tanh(conv1(X))
        X = self.lrelu(conv2(X))
        return torch.squeeze(self.pooling_1(X), dim=4)

    def _norm_conv(self, conv, X):
        if self.normalization_type == 'layernorm':
            return self.lrelu(self.normalization(conv(X).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return self.lrelu(self.normalization(conv(X)))

    def forward(self, X):
        X = self.dropout(X)
        X = torch.unsqueeze(X, dim=1)
        acc_data  = X[:, :, :, :self.r, :]
        gyro_data = X[:, :, :, self.r:, :]

        acc_emb  = self._conv_module(acc_data,  self.conv_1_1, self.conv_2_1)
        gyro_emb = self._conv_module(gyro_data, self.conv_1_2, self.conv_2_2)
        both_emb = self._conv_module(X,         self.conv_1_3, self.conv_2_3)

        X_fused = torch.cat((acc_emb, gyro_emb, both_emb), dim=3)
        X_fused = self.pooling_2(self._norm_conv(self.conv_3, X_fused))
        X_output = torch.squeeze(self._norm_conv(self.conv_4, X_fused), dim=3)

        return X_output.permute(0, 2, 1)


class WaveNet(Module):

    def __init__(self,
                 input_channels=140,
                 residual_channels=64,
                 postproc_channels=64,
                 output_channels=7,
                 dilations=[1, 2, 4],
                 filter_width=5,
                 conv_input_kernel_size=1,
                 conv_postproc_2_kernel_size=1,
                 conv_input_stride=1,
                 conv_postproc_1_stride=1,
                 conv_postproc_2_stride=1,
                 res_conv_stride=1,
                 conv_input_zero_padding='same',
                 conv_postproc_1_zero_padding='same',
                 conv_postproc_2_zero_padding='same',
                 res_conv_zero_padding='same',
                 dropout=0.3):

        super().__init__()

        self.residual_channels = residual_channels

        self.conv_input = Conv1d(input_channels, residual_channels,
                                 conv_input_kernel_size, conv_input_stride,
                                 conv_input_zero_padding)

        self.residual_convolutions = ModuleList([
            Conv1d(residual_channels, 2 * residual_channels,
                   filter_width, res_conv_stride, res_conv_zero_padding,
                   dilation=dilations[i])
            for i in range(len(dilations))
        ])

        self.conv_postproc_1 = Conv1d(residual_channels, postproc_channels,
                                      filter_width, conv_postproc_1_stride,
                                      conv_postproc_1_zero_padding)

        self.conv_postproc_2 = Conv1d(postproc_channels, output_channels,
                                      conv_postproc_2_kernel_size, conv_postproc_2_stride,
                                      conv_postproc_2_zero_padding)

        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, X):
        # X: (B, Nframes, input_channels)
        skip_outputs = []
        X = self.dropout(X)
        X = X.permute(0, 2, 1)  # (B, input_channels, Nframes)

        X = self.tanh(self.conv_input(X))

        for conv in self.residual_convolutions:
            res_input = X
            X = conv(res_input)
            X = self.tanh(X[:, :self.residual_channels, :]) * self.sigmoid(X[:, self.residual_channels:, :])
            skip_outputs.append(X)
            X += res_input

        Y = sum(skip_outputs)

        Encoding = self.relu(self.conv_postproc_1(Y))
        X_output = self.conv_postproc_2(Encoding)

        return X_output.permute(0, 2, 1), Encoding.permute(0, 2, 1)
