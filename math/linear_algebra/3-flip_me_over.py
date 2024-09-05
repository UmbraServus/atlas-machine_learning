#!/usr/bin/env python3
""" module for transposing a 2d matrix """


def matrix_transpose(matrix):
    """ function for transposing a 2d matrix """
    matrixT = []

    for col_index in range(len(matrix[0])):
        new_row = []

        for row_index in range(len(matrix)):
            new_row.append(matrix[row_index][col_index])
        matrixT.append(new_row)
    return matrixT
