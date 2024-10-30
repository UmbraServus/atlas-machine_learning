#!/usr/bin/env python3
"""module for lenet5 using v1 tensorflow"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a mod ver of the LeNet-5 architecture using tensorflow
args:
    x: tf.placeholder shape (m, 28, 28, 1) containing input images
        m is the number of images
    y: tf.placeholder shape (m, 10) containing the one-hot labels

The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

All layers requiring initialization should initialize their kernels with
the he_normal initialization 
method: tf.keras.initializers.VarianceScaling(scale=2.0)

All hidden layers requiring activation should use the relu activation
function

Returns:
tensor for the softmax activated output
training op that utilizes Adam opt (with default hyperparameters)
tensor for the loss of the netowrk
tensor for the accuracy of the network"""
