#!/usr/bin/env python3
""" module documentation """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a neural network using
    batch normalization:
    args:
        Z: a numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma: a np.ndarray shape (1, n) containing the scales used
        for batch normalization
        beta: a np.ndarray shape (1, n) containing the offsets used
        for batch normalization
        epsilon: a small number used to avoid division by zero
    Returns: the normalized Z matrix"""
