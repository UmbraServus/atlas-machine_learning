#!/usr/bin/env python3
"""module for L2 regularization."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ that calculates the cost of a neural network with L2 regularization:
    args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases (numpy.ndarrays) of the neural network
        L: number of layers in the neural network
        m: number of data points used
    Returns: the cost of the network accounting for L2 regularization"""

    L2 = 0
    for i in range(1, L + 1):
        W = weights[f'W{i}']
        L2 += np.sum(np.square(W))
    T_cost = cost + (lambtha / (2 * m)) * L2
    return T_cost
