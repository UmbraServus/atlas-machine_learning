#!/usr/bin/env python3
"""module intra cluster variance"""
import numpy as np


def variance(X, C):
    """ method that calculates the total intra-cluster variance for a data set
    X: a numpy.ndarray of shape (n, d) containing the data set
    C: a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
Returns: var, or None on failure
    var is the total variance """
