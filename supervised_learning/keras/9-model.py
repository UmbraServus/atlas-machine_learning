#!/usr/bin/env python3
"""module for saving and loading a model"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire model
    args:
        network: the model to save
        filename: the path of the file that the model should be saved to
    Returns: None """

    network.save(filename)
    return None


def load_model(filename):
    """ loads an entire model:
    args:
        filename: the path of the file that the model should be loaded from
    Returns: the loaded model """

    model = K.models.load_model(filename)
    return model
