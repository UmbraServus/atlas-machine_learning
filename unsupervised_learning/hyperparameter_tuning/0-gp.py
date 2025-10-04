#!/usr/bin/env python3
""" 0. Initialize Gaussian Process """
import numpy as np


class GaussianProcess:
    """ class that represents a noiseless 1D Gaussian Process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices

        Parameters:
        X1: numpy.ndarray of shape (m, d)
        X2: numpy.ndarray of shape (n, d)

        Returns:
        K: numpy.ndarray of shape (m, n)
        """
        diff = X1[:, np.newaxis] - X2.T
        sqdist = np.sum(np.square(diff), axis=1)
        K = self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))
        return K
