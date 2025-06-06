#!/usr/bin/env python3
"""moodule finding optimum n of clusters by var."""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """that tests the optimum number of clusters by variance:

    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer containing the minimum number of clusters to check
    (inclusive)
    kmax: positive integer containing the maximum number of clusters to check
    (inclusive)
    iterations: positive integer containing the maximum number of iterations
    K-means
        This function should analyze at least 2 different cluster sizes

    You should use:
        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance
    You may use at most 2 loops

    Returns: results, d_vars, or None, None on failure
        results: list containing the outputs of K-means each cluster size
        d_vars is a list containing the difference in variance from the
        smallest
        cluster size each cluster size"""
