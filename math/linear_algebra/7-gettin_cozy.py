#!/usr/bin/env python3
""" module for concat 2D matrices along an axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concats two 2Dmatrices along an axis if given """

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_matrix = [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
    return new_matrix
