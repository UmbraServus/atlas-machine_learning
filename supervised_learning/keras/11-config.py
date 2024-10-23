#!/usr/bin/env python3
"""module for serializing and deserializing in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s configuration in JSON format
    args:
        network: the model whose configuration should be saved
        filename: the path of the file that the configuration should be saved to
    Returns: None """
    config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(config)

def load_config(filename):
    """loads a model with a specific configuration:
    args:
        filename: the path of the file containing the model’s configuration in JSON format
    Returns: the loaded model"""

    with open(filename, 'r') as json_file:
        config = json_file.read()

    model = K.models.model_from_json(config)
    return model