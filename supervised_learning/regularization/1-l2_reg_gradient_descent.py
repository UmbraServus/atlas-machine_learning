#!/usr/bin/env python3
""" module for L2 reg w gradient descent """
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights & biases of a neural network usin grad desc w/ L2
    args:
        Y: one-hot np.ndarr shape (classes, m) tht contains the corr. labels
            classes is the number of classes
            m is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of ea. layer of the neural network
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers of the network """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        prev_A = cache[f"A{i-1}"]
        dW = (1 / m) * np.dot(dZ, prev_A.T) + (lambtha / m) * weights[f"W{i}"]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dZ = (
            np.dot(weights[f"W{i}"].T, dZ)
            * (prev_A * (1 - prev_A))
                )
        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
