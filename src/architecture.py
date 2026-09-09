'''architecture.py
this file contains the architecture of our neural network
composed of CNN  and MLP layers
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave donkeng ndia
version: 1.0
'''
import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """
    ''' Define the layers of your network in the constructor.
        we have two hidden layers in order to increase
        the learning capacity of the network
    '''
    '''
        we used mixed MPL && CNN architecture with 2
        hidden layers and the advantage of this is that
        it can capture both spatial features
        (using CNN) and complex patterns (using MLP)
    '''

    def __init__(
            self,
            input_channels,
            conv_hidden1,
            conv_hidden2,
            mlp_hidden1,
            mlp_hidden2,
            output_size):
        '''
    Args:
        input_channels: number of channels in input tensor (from data.py)
        conv_hidden1: number of filters in first conv layer
        conv_hidden2: number of filters in second/third conv layers
        mlp_hidden1: number of units in first dense layer
        mlp_hidden2: number of units in second dense layer
        output_size: number of possible actions
        (typically 5: North, South, East, West, Stop)
        '''

        super().__init__()
        # Store parameters for use in forward pass
        self.input_channels = input_channels
        self.conv_hidden1 = conv_hidden1
        self.conv_hidden2 = conv_hidden2
        self.mlp_hidden1 = mlp_hidden1
        self.mlp_hidden2 = mlp_hidden2
        self.output_size = output_size

        # ----------- CNN -----------
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, conv_hidden1, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_hidden1),  # normalize activations
            nn.ReLU(),  # introduces non-linearity
            nn.Dropout2d(0.1),  # prevents overfitting

            # Layer 2
            nn.Conv2d(conv_hidden1, conv_hidden2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_hidden2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),  # downsample feature maps

            # Layer 3
            nn.Conv2d(conv_hidden2, conv_hidden2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_hidden2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
        )

        # adaptive pooling to get fixed size output (conv_hidden2 x 1 x 1)
        self.adapt = nn.AdaptiveAvgPool2d((1, 1))

        # ----------- MLP -----------
        self.mlp = nn.Sequential(
            # learns complex patterns from extracted CNN features
            nn.Linear(conv_hidden2, mlp_hidden1),
            nn.ReLU(),  # introduces non-linearity
            nn.Dropout(0.3),  # prevents overfitting

            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),  # prevents overfitting


            # outputs Q-values for each action
            nn.Linear(mlp_hidden2, output_size)
        )

    def forward(self, x):
        x = self.cnn(x)  # pass input through CNN layers
        x = self.adapt(x)  # adaptive pooling

        x = torch.flatten(x, 1)  # flatten for MLP input
        x = self.mlp(x)
        return x
