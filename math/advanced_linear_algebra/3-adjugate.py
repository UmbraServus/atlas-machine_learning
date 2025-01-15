#!/usr/bin/env python3
"""module for calc the adjugate matrix (cofactor transpose)"""
cofactor = __import__('2-cofactor').cofactor


def matrix_transpose(matrix):
    """ function for transposing a 2d matrix """
    matrixT = []

    for col_index in range(len(matrix[0])):
        new_row = []

        for row_index in range(len(matrix)):
            new_row.append(matrix[row_index][col_index])
        matrixT.append(new_row)
    return matrixT


def adjugate(matrix):
    """ that calculates the adjugate matrix of a matrix
matrix: list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
    message matrix must be a non-empty square matrix
Returns: the adjugate matrix of matrix"""

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    cofac_mat = cofactor(matrix)

    adjugate_mat = matrix_transpose(cofac_mat)

    return adjugate_mat
