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
    You are not allowed to use the Input class
    Returns: the keras model"""

    model = K.Sequential()
    
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha),
                input_shape=(nx,)
                ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha),
            ))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))
    return model
