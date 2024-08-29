#!/usr/bin/env python3
""" module for calculating shape of matrix """

def matrix_shape(matrix):
    """ function for matrix shape calc """

    if not matrix:
        return ()
    
    if isinstance(matrix[0], list):
        if isinstance(matrix[0][0], list):
            matrices = len(matrix)
            rows = len(matrix[0])
            columns = len(matrix[0][0])
            return matrices, rows, columns
        else:
            rows = len(matrix)
            columns = len(matrix[0]) if rows > 0 else 0
            return rows, columns
    else:
        return len(matrix)