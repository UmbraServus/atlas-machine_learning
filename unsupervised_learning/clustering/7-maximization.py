#!/usr/bin/env python3
"""module calc the maximization step in EM alg in a gmm"""
import numpy as np


def maximization(X, g):
    """that calculates the maximization step in the EM algorithm a GMM

X is a numpy.ndarray of shape (n, d) containing the data set
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
each data point in each cluster

You may use at most 1 loop

Returns: pi, m, S, or None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the updated priors each
cluster
m is a numpy.ndarray of shape (k, d) containing the updated centroid means
each cluster
S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
matrices each cluster"""
