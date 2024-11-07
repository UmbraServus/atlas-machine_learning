#!/usr/bin/env python3
"""module that builds a transition layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """that builds a transition layer as described in
Densely Connected Convolutional Networks:
args:
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively
Returns: The output of the transition layer and
the number of filters within the output, respectively"""
