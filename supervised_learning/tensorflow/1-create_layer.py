#!/usr/bin/env python3
""" module for function create layer """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """ method creates_layer that initialzes and creates a layer in the
    neural network
        args:
            prev: the tensor output of the previous layer
            n: the number of nodes in the layer to create
            activation: the activation function that the layer should use
        return: tensor output of the layer
            """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    return layer(prev)

