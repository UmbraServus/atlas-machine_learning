#!/usr/bin/env python3
"""module kmeans"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """method performs K-means on a dataset:

X: numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions each data point
k: positive integer containing the number of clusters
iterations: a positive integer containing the maximum number of iterations
that should be performed

If no change in the cluster centroids occurs between iterations, your
function should return
Initialize the cluster centroids using a multivariate uniform distribution
(based on0-initialize.py)
If a cluster contains no data points during the update step, reinitialize its
centroid
You should use numpy.random.uniform exactly twice
You may use at most 2 loops
Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to"""
