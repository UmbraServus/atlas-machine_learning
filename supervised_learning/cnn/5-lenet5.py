#!/usr/bin/env python3
"""module for lenet5 using keras"""
from tensorflow import keras as K


def lenet5(X):
    """ builds a modified ver. of the LeNet-5 architecture using keras
args:
    X: a K.Input of shape (m, 28, 28, 1) containing the input images
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
    initializer = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120,
                         kernel_initializer=initializer,
                         activation='relu')(flatten)
    fc2 = K.layers.Dense(units=84,
                         kernel_initializer=initializer,
                         activation='relu')(fc1)
    outputs = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=initializer)(fc2)

    model = K.Model(inputs=X, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
