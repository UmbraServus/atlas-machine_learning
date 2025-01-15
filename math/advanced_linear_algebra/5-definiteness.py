#!/usr/bin/env python3
"""module for calc definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """that calculates the definiteness of a matrix:

matrix: np.ndarr shape (n, n) whose definiteness should be calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message
    matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None

Return: the string Positive definite,
Positive semi-definite,
Negative semi-definite,
Negative definite, or
Indefinite if the matrix is positive definite,
positive semi-definite,
negative semi-definite,
negative definite of indefinite, respectively
If matrix does not fit any of the above categories, return None
"""
