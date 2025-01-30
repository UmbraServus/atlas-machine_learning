#!/usr/bin/env python3
"""module bayesian inmation criterion """
import numpy as np
expectation_maximization =__import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ method that finds the best number of clusters a GMM using the
    Bayesian Infrmation Criterion:

    X: a numpy.ndarray of shape (n, d) containing the data set

    kmin: a positive integer containing the minimum number of clusters
    to check (inclusive)

    kmax: a positive integer containing the maximum number of clusters
    to check  (inclusive)
        If kmax is None, kmax should be set to the maximum
        number of clusters possible

    iterations: a positive integer containing the maximum number of
    iterations the EM algorithm

    tol: a non-negative float containing the tolerance  the EM algorithm

    verbose: a boolean that determines if the EM algorithm should print
    inmation to the standard output

    You may use at most 1 loop

    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value  k based on its BIC
        best_result is tuple containing pi, m, S

        pi is a numpy.ndarray of shape (k,) containing the cluster priors
         the best number of clusters
        m is a numpy.ndarray of shape (k, d) containing the centroid means
         the best number of clusters
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices  the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood  each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value  each cluster size tested

        Use: BIC = p * ln(n) - 2 * l

        p is the number of parameters required  the model
        n is the number of data points used to create the model
        l is the log likelihood of the model"""
