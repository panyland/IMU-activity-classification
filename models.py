import numpy as np
import math
import time
import sys
import torch
from torch.nn import Module, ModuleList, Linear, Conv1d, BatchNorm1d, Dropout, GELU, LayerNorm, Parameter, Identity, ELU
from torch.nn import AvgPool1d, Conv2d, AvgPool2d, LeakyReLU, Tanh, Sigmoid, ELU, Conv3d, AvgPool3d, BatchNorm2d
from torch.nn import ReLU, MaxPool1d


class CNN_encoder(Module):

    def __init__(self,
                 s_channels = 18,
                 input_channels = 120,
                 latent_channels = 70,
                 output_channels = 140,
                 dropout = 0.3,
                 conv_1_kernel_size = (1,3,11),
                 conv_2_kernel_size = (1,3,5),
                 conv_3_kernel_size = (1,3),
                 conv_4_kernel_size = (1,4), 
                 conv_1_stride = (1,3,6),
                 conv_2_stride = (1,1,2),
                 conv_3_stride = 1,
                 conv_4_stride = 1,
                 conv_1_zero_padding = (0,0,0),
                 conv_2_zero_padding = (0,1,2),
                 conv_3_zero_padding = 'same',
                 conv_4_zero_padding = 'valid',
                 pooling_1_zero_padding = (0,0,0),
                 pooling_1_kernel_size = (1,1,10),
                 pooling_2_zero_padding = (0,0),
                 pooling_2_kernel_size = (1,3),
                 normalization_type = 'layernorm'):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.r = np.int32(s_channels / 2) # Default: 12
        
        self.conv_1_1 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        self.conv_1_2 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        self.conv_1_3 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        
        self.conv_2_1 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        self.conv_2_2 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        self.conv_2_3 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        
        self.conv_3 = Conv2d(in_channels=latent_channels, out_channels=output_channels, 
                             kernel_size=conv_3_kernel_size, stride=conv_3_stride,
                             padding=conv_3_zero_padding, bias=True)
        
        self.conv_4 = Conv2d(in_channels=output_channels, out_channels=output_channels, 
                             kernel_size=conv_4_kernel_size, stride=conv_4_stride,
                             padding=conv_4_zero_padding, bias=True)
        
        self.pooling_1 = AvgPool3d(kernel_size=pooling_1_kernel_size, padding=pooling_1_zero_padding)
        self.pooling_2 = AvgPool2d(kernel_size=pooling_2_kernel_size, padding=pooling_2_zero_padding)
        
        self.normalization = normalization_layer(output_channels)
        self.normalization_type = normalization_type
        
        self.tanh = Tanh()
        self.lrelu = LeakyReLU()
        
        self.dropout = Dropout(dropout)


    def _conv_module_1(self, X):
        
        # X is now of shape [batch_size, 1, Nframes, 3*x, 120]
        X = self.tanh(self.conv_1_1(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_1(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, x]
        
        return X
    
    def _conv_module_2(self, X):
        
        # X is now of shape [Nframes, 1, 3*x, 120]
        X = self.tanh(self.conv_1_2(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_2(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, x]
        
        return X
    
    def _conv_module_3(self, X):
        
        # X is now of shape [Nframes, 1, 6*x, wl]
        X = self.tanh(self.conv_1_3(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_3(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, 2*x]
        
        return X
    
    def forward(self, X):
        
        X = self.dropout(X) # X is of size [batch_size, Nframes, s_channels, input_channels]
        X = torch.unsqueeze(X, dim=1) # X is of size [batch_size, 1, Nframes, s_channels, input_channels]
        acc_data = X[:, :, :, :(self.r), :] # acc_data is of size [batch_size, 1, Nframes, r, input_channels]
        gyro_data = X[:, :, :, (self.r):, :] # gyro_data is of size [batch_size, 1, Nframes, r, input_channels]
        
        # Get convolution embeddings for acceleration and gyro, both separately and together
        acc_emb = self._conv_module_1(acc_data) # acc_emb is of size [batch_size, latent_channels, Nframes, x]
        gyro_emb = self._conv_module_2(gyro_data) # gyro_emb is of size [batch_size, latent_channels, Nframes, x]
        both_emb = self._conv_module_3(X) # X is of size [batch_size, latent_channels, Nframes, 8]
        
        # We fuse acceleration and gyro
        X_fused = torch.cat((acc_emb, gyro_emb, both_emb), dim=3) # X_fused is of size [batch_size, latent_channels, Nframes, 16]
        if self.normalization_type == 'layernorm':
            X_fused = self.pooling_2(self.lrelu(self.normalization(self.conv_3(X_fused).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        else:
            X_fused = self.pooling_2(self.lrelu(self.normalization(self.conv_3(X_fused))))
        
        # We fuse sensors
        if self.normalization_type == 'layernorm':
            X_output = torch.squeeze(self.lrelu(self.normalization(self.conv_4(X_fused).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)), dim=3)
        else:
            X_output = torch.squeeze(self.lrelu(self.normalization(self.conv_4(X_fused))), dim=3)
        
        X_output = X_output.permute(0, 2, 1) # X_output is of size [batch_size, Nframes, output_channels]   
        
        return X_output


class final_linear_decoder(Module):
    """
    Used to map waveNet output to class probabilities for activity classification? Maybe more layers?
    
    """
    
    def __init__(self,
                 input_dim = 128,
                 output_dim = 11):

        super().__init__()
        
        self.linear_layer = Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, X):
        
        # X is now of size [batch_size, num_frames_embedding, num_features_embedding]
        X_output = self.linear_layer(X)
        
        # X_output is of size [batch_size, num_frames_embedding, output_dim]
        
        return X_output


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

        self.conv_input = Conv1d(in_channels=input_channels, out_channels=residual_channels, 
                                 kernel_size=conv_input_kernel_size, stride=conv_input_stride,
                                 padding=conv_input_zero_padding, bias=True)

        self.residual_convolutions = ModuleList([Conv1d(in_channels=residual_channels,
                                                        out_channels=2*residual_channels,
                                                        kernel_size=filter_width, stride=res_conv_stride,
                                                        padding=res_conv_zero_padding, dilation=dilations[i],
                                                        bias=True) for i in range(len(dilations))])

        self.conv_postproc_1 = Conv1d(in_channels=residual_channels, out_channels=postproc_channels, 
                                      kernel_size=filter_width, stride=conv_postproc_1_stride,
                                      padding=conv_postproc_1_zero_padding, bias=True)

        self.conv_postproc_2 = Conv1d(in_channels=postproc_channels, out_channels=output_channels, 
                                      kernel_size=conv_postproc_2_kernel_size, stride=conv_postproc_2_stride,
                                      padding=conv_postproc_2_zero_padding, bias=True)

        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

        self.dropout = Dropout(dropout)

    def forward(self, X):

        skip_outputs = []
        X = self.dropout(X)  # X is of size [Nframes, input_channels]
        X = torch.unsqueeze(X, dim=2).permute(2, 1, 0)  # X is of size [1, input_channels, Nframes]

        # Input convolution
        X = self.tanh(self.conv_input(X))  # X is of size [1, residual_channels, Nframes]

        # Perform the dilated convolutions with filtering, gating, and residual connections
        for i in range(len(self.residual_convolutions)):
            res_input = X

            # Dilated convolutions over time
            X = self.residual_convolutions[i](res_input)

            # Filter and gate
            X = self.tanh(X[:, :(self.residual_channels), :]) * self.sigmoid(X[:, (self.residual_channels):, :])

            skip_outputs.append(X)
            X += res_input

        Y = sum(skip_outputs)  # Y is of size [1, residual_channels, Nframes]

        # Post-processing
        Encoding = self.relu(self.conv_postproc_1(Y))
        X_output = self.conv_postproc_2(Encoding)
        Encoding = torch.squeeze(Encoding).permute(1, 0)  # Encoding is of size [Nframes, postproc_channels]
        X_output = torch.squeeze(X_output).permute(1, 0)  # X_output is of size [Nframes, output_channels]

        return X_output, Encoding