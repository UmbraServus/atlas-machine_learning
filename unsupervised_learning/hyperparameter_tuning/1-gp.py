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

    def predict(self, X_s):
        """ predicts the mean and std dev of pts in a Gaussian process
    X_s is a numpy.ndarray of shape (s, 1) containing all of the points whose
    mean and standard deviation should be calculated
            s is the number of sample points
    Returns: mu, sigma"""
        # Calc. the covariance matrix betwn the training pts and the new pts
        K_s = self.kernel(self.X, X_s)
        # Calculate the covariance matrix between the new points
        K_ss = self.kernel(X_s, X_s)
        # Calculate the inverse of the covariance matrix of the training pts
        K_inv = np.linalg.inv(self.K)
        # Calculate the mean of the new points
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        # Calculate the covariance of the new points
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        # Calculate the standard deviation of the new points
        sigma = np.diag(cov)
        # Return the mean and the std. dev. (sqrt of the covariance diagonal)
        return mu, sigma
