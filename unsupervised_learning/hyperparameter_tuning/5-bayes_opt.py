#!/usr/bin/env python3
"""bayes opt class module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor.

        Parameters:
        f: the black-box function to be optimized
        X_init: numpy.ndarray of shape (t, 1) representing the inputs already
        sampled with the black-box function
        Y_init: numpy.ndarray of shape (t, 1) representing the outputs of the
        black-box function for each input in X_init
        t: the number of initial samples
        bounds: tuple of (min, max) representing the bounds of the space in
        which to look for the optimal point
        ac_samples: the number of samples that should be analyzed during
        acquisition
        l: the length parameter for the kernel
        sigma_f: the standard deviation given to the output of the black-box
        function
        xsi: the exploration-exploitation factor for acquisition
        minimize: bool determining whether optimization should be performed
        for minimization (True) or maximization (False)

        Sets the following public instance attributes:
        f: the black-box function
        gp: an instance of the class GaussianProcess
        X_s: numpy.ndarray of shape (ac_samples, 1) containing all acquisition
        sample points, evenly spaced between min and max
        xsi: the exploration-exploitation factor
        minimize: bool for minimization versus maximization
        """
        # Your initialization code here
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected
        Improvement acquisition function.

        Returns:
        X_next: numpy.ndarray of shape (1,) representing the next
        best sample point
        EI: numpy.ndarray of shape (ac_samples,) containing the expected
        improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample = self.gp.predict(self.gp.X)
        sigma = np.maximum(sigma, 1e-9)
        EI = BayesianOptimization.compute_ei(mu, sigma, mu_sample,
                                             self.minimize, self.xsi)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """ Optimizes the black-box function.

            Parameters:
            iterations: the maximum number of iterations to perform

            Returns:
            X_opt: numpy.ndarray of shape (1,) representing the optimal point
            Y_opt: numpy.ndarray of shape (1,) representing the optimal
            function value"""

        for i in range(iterations):
            # Calculate the next sampling point using the acq function
            X_next, _ = self.acquisition()

            # If the next point is close to the sample, stop early
            if np.any(np.abs(X_next - self.gp.X) <= 1e-10):
                break
            # Sample the function at the new point
            Y_next = self.f(X_next)
            # Update the Gaussian process with the new sample
            self.gp.update(X_next, Y_next)
            # Determine the optimal point and its function value
        if self.minimize:
            idx = np.argmin(self.gp.Y)

        else:
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt

    @staticmethod
    def compute_ei(mu, sigma, mu_sample, minimize, xsi):
        """
        Computes the expected improvement of a sample point.

        Parameters:
        mu: numpy.ndarray of shape (ac_samples,) containing the mean
        of the GP
        sigma: numpy.ndarray of shape (ac_samples,) containing the std
        dev of the GP
        mu_sample: numpy.ndarray of shape (t, 1) containing the means
        of sampled pts
        minimize: bool for minimization (True) or maximization (False)
        xsi: the exploration-exploitation factor for acquisition

        Returns:
        EI: numpy.ndarray of shape (ac_sample   s,) containing
        the expected improvement
        """
        if minimize:
            mu_sample_opt = np.min(mu_sample)
            improvement = mu_sample_opt - mu - xsi
        else:
            mu_sample_opt = np.max(mu_sample)
            improvement = mu - mu_sample_opt - xsi
        with np.errstate(divide='ignore'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0] = 0
        return EI
