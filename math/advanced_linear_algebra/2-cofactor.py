#!/usr/bin/env python3
"""module for cofactor calc"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """that calculates the cofactor matrix of a matrix:
matrix: a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

Returns: the cofactor matrix of matrix"""

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for row in range(len(minor_matrix)):
        cofactor_row = []
        for col in range(len(minor_matrix[row])):
            cofactor_value = minor_matrix[row][col] * ((-1) ** (row + col))
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix