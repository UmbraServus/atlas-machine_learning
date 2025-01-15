#!/usr/bin/env python3
"""module for cal correlation"""
import numpy as np

def correlation(C):
    """calculates a correlation matrix
    cor = C / stddev(xi) * stddev(xj)    
    C: np.ndarr shape (d, d) containing a covariance matrix
        d is the number of dimensions

        If C is not a numpy.ndarray, raise a TypeError with the
        message C must be a numpy.ndarray
        If C does not have shape (d, d), raise a ValueError with the
        message C must be a 2D square matrix

    Returns: np.ndarr shape (d, d) containing the correlation matrix"""

    if not isinstance(C, np.ndarray) or C.shape != 2:
        raise TypeError("C must be a numpy.ndarray")

    d, d2 = C.shape

    if d != d2:
        raise ValueError("C must be a 2D square matrix")

    std_dev = np.sqrt(np.diag(C))
    std_dev_mat = std_dev[:, np.newaxis] * std_dev[np.newaxis, :]
    cor_mat = C / std_dev_mat

    return cor_mat
