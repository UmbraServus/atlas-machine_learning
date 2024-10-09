#!/usr/bin/env python3
"""module to calculate normalization constants of a matrix."""
import numpy as np


def normalization_constants(X):
    """that calcs the normalization (stndrdization) constants of a matrix
    X: the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    Returns: mean and stndrd deviation of ea feature, respectively """
    mean = np.mean(X, axis=0)
    stdd = np.std(X, axis=0)
    return mean, stdd
