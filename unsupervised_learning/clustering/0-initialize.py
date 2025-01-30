#!/usr/bin/env python3
"""module  initializing k-cluster centroids."""
import numpy as np


def initialize(X, k):
    """method that initializes cluster centroids  K-means:

X: numpy.ndarray of shape (n, d) containing the dataset that will be used
K-means clustering
    n is the number of data points
    d is the number of dimensions  each data point
k: a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate unim
    distribution along each dimension in d:
    The minimum values  the distribution should be the minimum values of
    X along each dimension in d
    The maximum values  the distribution should be the maximum values of
    X along each dimension in d
You should use numpy.random.unim exactly once
    You are not allowed to use any loops
Returns: a numpy.ndarray of shape (k, d) containing the initialized centroids
 each cluster, or None on failure"""
