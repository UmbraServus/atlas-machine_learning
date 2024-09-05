#!/usr/bin/env python3
""" module for concat matrices along axis with np. """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ method for using np.concat """
    return np.concatenate((mat1, mat2), axis)
