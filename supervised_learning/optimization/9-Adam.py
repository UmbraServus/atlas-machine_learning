#!/usr/bin/env python3
""" module documentation """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm:
    args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
        var: a numpy.ndarray containing the variable to be updated
        grad: a numpy.ndarray containing the gradient of var
        v: the previous first moment of var
        s: the previous second moment of var
        t: the time step used for bias correction
    Returns: updated var, new 1st moment, and new 2nd moment, respectively"""