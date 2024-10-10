#!/usr/bin/env python3
"""module for updating a variable using grad. descent w// momentum opti alg."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updts a var. usin the gradient descent w momentum opti. algorithm
    v = β1 * v + (1 − β1) * ∇J(θ)
    var = var - alpha * v
    args:
        alpha is the learning rate
        beta1 is the momentum weight
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        v is the previous first moment of var
    Returns: the updated variable and the new moment, respectively"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v