#!/usr/bin/env python3
"""module for predict with keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """that makes a prediction using a neural network
args:
    network: network model to make the prediction with
    data: input data to make the prediction with
    verbose: boolean that determines if output should be printed
    during the prediction process
Returns: the prediction for the data"""
    return network.predict(data, verbose=verbose)
