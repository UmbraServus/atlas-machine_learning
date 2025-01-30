#!/usr/bin/env python3
""" module  gaussian mixture model intialization."""
import numpy as np


def initialize(X, k):
    """that initializes variables  a Gaussian Mixture Model:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
        You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors each
        cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid means
         each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices  each cluster, initialized as identity matrices

    You should use kmeans = __import__('1-kmeans').kmeans"""
