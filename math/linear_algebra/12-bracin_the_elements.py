#!/usr/bin/env python3
"""  module for doing simple arithmetic of
2 matrices by the elements in a method """


def np_elementwise(mat1, mat2):
    """ method for doing simple arithmetic of 2 matrices by the elements """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
