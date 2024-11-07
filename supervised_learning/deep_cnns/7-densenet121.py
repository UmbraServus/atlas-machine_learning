#!/usr/bin/env python3
"""module that builds a densenet121"""
from tensorflow import keras as K


def densenet121(growth_rate=32, compression=1.0):
    """that builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks:
args:
    growth_rate is the growth rate
    compression is the compression factor

    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
You may use:
    dense_block = __import__('5-dense_block').dense_block
    transition_layer = __import__('6-transition_layer').transition_layer
Returns: the keras model"""
