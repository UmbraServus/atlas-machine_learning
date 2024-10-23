#!/usr/bin/env python3
"""module for saving the weights of a model manually"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ saves a model’s weights
    args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved
    Returns: None"""
    network.save_weights(filename, save_format=save_format)
    return None

def load_weights(network, filename):
    """ loads a model’s weights
    args:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from
    Returns: None """

    network.load_weights(filename)
    return None
