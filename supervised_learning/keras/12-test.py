#!/usr/bin/env python3
""" module to test/evaluate a model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ that tests a neural network
    args:
        network: the network model to test
        data: the input data to test the model with
        labels: the correct one-hot labels of data
        verbose: bool tht determines if output shld be prntd durin the testin
    Returns: loss and accuracy of the model w/ the testin data, respectively
    """
    test_loss, test_acc = network.evaluate(data, labels, verbose=verbose)
    return test_loss, test_acc
