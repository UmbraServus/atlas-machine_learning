#!/usr/bin/env python3
""" module that builds an inception block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """method that builds an inception block
    args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP
            F1 number of filters in the 1x1 convolution
            F3R number of filters in the 1x1 conv b4 the 3x3 convolution
            F3 number of filters in the 3x3 convolution
            F5R number of filters in the 1x1 conv. before the 5x5 convolution
            F5 number of filters in the 5x5 convolution
            FPP number of filters in the 1x1 convolution aftr the max pooling
    All convolutions inside the inception block should use
    rectified linear activation (ReLU)
Returns: the concatenated output of the inception block """

    F1, F3R, F3, F5R, F5, FPP = filters
    branch_1 = K.layers.Conv2D(filters=F1,
                               kernel_size=1,
                               activation='relu')(A_prev)

    branch_2 = K.layers.Conv2D(filters=F3R,
                               kernel_size=1,
                               activation='relu',
                               padding='same')(A_prev)
    branch_2 = K.layers.Conv2D(filters=F3,
                               kernel_size=3,
                               activation='relu',
                               padding='same')(branch_2)

    branch_3 = K.layers.Conv2D(filters=F5R,
                               kernel_size=1,
                               activation='relu',
                               padding='same')(A_prev)
    branch_3 = K.layers.Conv2D(filters=F5,
                               kernel_size=5,
                               activation='relu',
                               padding='same')(branch_3)

    branch_pool = K.layers.MaxPooling2D(pool_size=3,
                                        strides=1,
                                        padding='same')(A_prev)
    branch_pool = K.layers.Conv2D(filters=FPP,
                                  kernel_size=1,
                                  activation='relu')(branch_pool)

    output = K.layers.Concatenate(axis=-1)([branch_1,
                                            branch_2,
                                            branch_3,
                                            branch_pool])
    return output
