#!/usr/bin/env python3
"""  module for building a NN with Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library
    args:
        nx: number of input features to the network
        layers: list containing the # of nodes in ea layer of the network
        activations: list containing act. functs used for ea layer of network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns: the keras model"""

    inputs = K.Input(shape=(nx, ))

    for i in range(len(layers)):
        x = K.layers.Dense(64, activation=activations[i])(inputs)
        if i < len(layers) - 1:
            x = K.layers.Dropout(keep_prob)(x)
        outputs = K.layers.Dense(1, activation=activations[i])
    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
