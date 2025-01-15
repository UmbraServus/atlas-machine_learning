#!/usr/bin/env python3
"""module for inverse of a matrix"""
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """ that calculates the inverse of a matrix
matrix: a list of lists whose inverse should be calculated
    
    If matrix is not a list of lists, raise a TypeError with the 
    message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the 
    message matrix must be a non-empty square matrix

Returns: the inverse of matrix, or None if matrix is singular"""

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj_mat = adjugate(matrix)
    inverse_mat = [[adj_mat[r][c] * 1 / det for c in range(len(adj_mat[r]))
                    ] for r in range(len(adj_mat))]
    return inverse_mat
