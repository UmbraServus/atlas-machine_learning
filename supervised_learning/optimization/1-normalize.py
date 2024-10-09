#!/usr/bin/env python3
"""module to normalize a matrix"""


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix:
        X: the numpy.ndarray of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m: np.ndarr shape (nx,) that contains the mean of all features of X
        s: np.ndarr shape (nx,) that contains the stdrd devia. of all ft. of X
        Returns: The normalized X matrix"""
    return (X - m) / s
