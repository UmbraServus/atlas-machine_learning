#!/usr/bin/env python3
"""module pca and returning the weights W"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset:

X: a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
var: fraction of the variance that the PCA transformation should maintain

Returns: weights matrix, W, that maintains var fraction of X‘s orig variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
    of the transformed X"""

    cov_matrix = np.cov(X.T)
    eigval, eigvec = np.linalg.eig(cov_matrix)
    sorted_idxs = np.argsort(eigval)[::-1]
    sorted_eigval = eigval[sorted_idxs]
    sorted_eigvec = eigvec[:, sorted_idxs]
    tot_var = np.sum(sorted_eigval)
    explained_var = np.cumsum(sorted_eigval) / tot_var
    nd = np.argmax(explained_var >= var) + 1
    W = sorted_eigvec[:, : nd + 1]
    for i in range(W.shape[1]):
        if W[0, i] < 0:
            W[:, i + 1] *= -1
    return W
