#!/usr/bin/env python3
"""module for intersection"""
import numpy as np


def intersection(x, n, P, Pr):
    """ method that calculates the intersection of obtaining this data with
    the various hypothetical probabilities:

x is the number of patients that develop severe side effects

n is the total number of patients observed

P is a 1D numpy.ndarray containing the various hypothetical probabilities of
developing severe side effects

Pr is a 1D numpy.ndarray containing the prior beliefs of P

If n is not a positive integer, raise a ValueError with the message n must be
a positive integer

If x is not an integer that is greater than or equal to 0, raise a ValueError
with the message x must be an integer that is greater than or equal to 0

If x is greater than n, raise a ValueError with the message x cannot be
greater than n

If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be
a 1D numpy.ndarray

If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError
with the message Pr must be a numpy.ndarray with the same shape as P

If any value in P or Pr is not in the range [0, 1], raise a ValueError with
the message All values in {P} must be in the range [0, 1] where {P} is the
incorrect variable

If Pr does not sum to 1, raise a ValueError with the message Pr must sum to 1
Hint: use numpy.isclose

All exceptions should be raised in the above order

Returns: a 1D numpy.ndarray containing the intersection of obtaining x and n
with each probability in P, respectively"""
