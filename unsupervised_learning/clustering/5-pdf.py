#!/usr/bin/env python3
"""module calc pdf of a gaussian distribution"""
import numpy as np


def pdf(X, m, S): 
    """that calculates the probability density function of a Gaussian
    distribution:

    X: a numpy.ndarray of shape (n, d) containing the data points whose
    PDF should be evaluated
    m: a numpy.ndarray of shape (d,) containing the mean of the distribution
    S: a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method
    numpy.ndarray.diagonal

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values each
        data point
        All values in P should have a minimum value of 1e-300"""
