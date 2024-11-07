#!/usr/bin/env python3
"""module that builds an identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """ method that builds an identity block from
    Deep Residual Learning for Image Recognition (2015)
args:
    A_prev: the output from the previous layer
    filters: tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution

        convs inside block should be followed by batch normalization along 
the channels axis and a rectified linear activation (ReLU), respectively.
        All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero

Returns: the activated output of the identity block"""
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    conv_1 = K.layers.Conv2D(filters=F11,
                             kernel_size=1,
                             kernel_initializer=initializer,
                             padding='same')(A_prev)
    normalization = K.layers.BatchNormalization(axis=-1)(conv_1)
    activation = K.layers.ReLU()(normalization)

    conv_2 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             kernel_initializer=initializer,
                             padding='same')(activation)
    normalization = K.layers.BatchNormalization(axis=-1)(conv_2)
    activation = K.layers.ReLU()(normalization)

    conv_3 = K.layers.Conv2D(filters=F12,
                             kernel_size=1,
                             kernel_initializer=initializer,
                             padding='same')(activation)
    normalization = K.layers.BatchNormalization(axis=-1)(conv_3)

    output = K.layers.Add()([normalization, A_prev])
    output = K.layers.ReLU()(output)

    return output
