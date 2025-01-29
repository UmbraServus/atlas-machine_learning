#!/usr/bin/env python3
"""module for pca V2"""
import numpy as np


def pca(X, ndim):
    """that performs PCA on a dataset:

X: a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point

ndim: the new dimensionality of the transformed X

Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
version of X"""

    n, d = X.shape
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.dot(X_centered.T, X_centered) / (n - 1)
    eigval, eigvec = np.linalg.eig(cov_matrix)
    sorted_idxs = np.argsort(eigval)[::-1]
    sorted_eigval = eigval[sorted_idxs]
    sorted_eigvec = eigvec[:, sorted_idxs]
    tot_var = np.sum(sorted_eigval)
    explained_var = np.cumsum(sorted_eigval) / tot_var
    W = sorted_eigvec[:, :ndim]
    for i in range(ndim):
        if W[0, i] < 0:
            W[:, i] *= -1
    T = np.dot(X_centered, W)
    return T
