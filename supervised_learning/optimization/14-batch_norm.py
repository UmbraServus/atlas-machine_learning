""" module documentation """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a neural network in tensorflow
    args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: function that should be used on the output of the layer
    Returns: a tensor of the activated output for the layer """
