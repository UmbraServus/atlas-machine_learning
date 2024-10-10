#!/usr/bin/env python3
""" module documentation"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the RMSProp optimization algorithm
    args:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
        var: a numpy.ndarray containing the variable to be updated
        grad: a numpy.ndarray containing the gradient of var
        s: the previous second moment of var
    Returns: the updated variable and the new moment, respectively"""
