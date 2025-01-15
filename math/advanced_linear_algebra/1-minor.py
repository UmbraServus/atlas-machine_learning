#!/usr/bin/env python3
"""module for minor matrix calc"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """method that calculates the minor matrix of a matrix
matrix: list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix
Returns: the minor matrix of matrix"""

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix") 
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    minor_mat = []
    for row in range(len(matrix)):
        minor_row = []
        for col in range(len(matrix[row])):
            sub_mat = [row[:col] + row[col + 1:] for row in (matrix[:row] + 
                                                             matrix[row+1:])]
            minor_row.append(determinant(sub_mat))
        minor_mat.append(minor_row)
    return minor_mat
