#!/usr/bin/env python3
"""module for lenet5 using keras"""
from tensorflow import keras as K


def lenet5(X):
    """ builds a modified ver. of the LeNet-5 architecture using keras
args:
    X: a K.Input of shape (m, 28, 28, 1) containing the input images for the network
        m is the number of images

The model should consist of the following layers in order:
    
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

All layers requiring initialization should initialize their kernels with
the he_normal initialization method
The seed for the he_normal initializer should be set to zero for each layer
to ensure reproducibility.
All hidden lyrs requiring activation should use the relu activation function
Returns: K.Model compiled 2 use Adam opt. (w/ def. hyperparas) & acc. metrics
"""
