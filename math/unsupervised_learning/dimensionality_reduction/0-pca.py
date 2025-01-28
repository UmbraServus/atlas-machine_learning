#!/usr/bin/env python3
import numpy as np
"""module for pca and returning the weights W"""


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
    nd = np.argmax(explained_var >= var) + 1
    W = sorted_eigvec[:,:nd]
    return W
