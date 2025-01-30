#!/usr/bin/env python3
"""module calcn expectation step in the EM algor of a GMM"""
import numpy as np


def expectation(X, pi, m, S):
    """ that calculates the expectation step in the EM algorithm a GMM:

    X: numpy.ndarray of shape (n, d) containing the data set
    pi: numpy.ndarray of shape (k,) containing the priors each cluster
    m: numpy.ndarray of shape (k, d) containing the centroid means each
    cluster
    S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    each cluster

    You may use at most 1 loop

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities each data point in each cluster
        l is the total log likelihood

    You should use pdf = __import__('5-pdf').pdf"""
