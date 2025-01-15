#!/usr/bin/env python3
"""module for mean and covariance"""
import numpy as np


def mean_cov(X):
    """that calculates the mean and covariance of a data set
    cov(X) = 1/(n-1)(X_centered.T)(X_centered)

    X: numpy.ndarray of shape (n, d) containing the data set
        n is the number of data points
        d is the number of dimensions in each data point

        If X is not a 2D numpy.ndarray, raise a TypeError with the
        message X must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the
        message X must contain multiple data points

    Returns: mean, cov:
        mean is np.ndarr shape (1, d) containing the mean of the data set
        cov is np.ndarr shape (d, d) containin the covar mat of the data set
    You are not allowed to use the function numpy.cov"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = np.shape(X)

    if n < 2:
        raise ValueError("X must contain muyltiple data points")

    mean = np.mean(X, axis=0)

    X_center = X - mean

    cov = np.dot(X_center.T, X_center) / (n - 1)

    return mean, cov
