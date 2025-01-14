#!/usr/bin/python3
""" calc determinant module """


def determinant(matrix):
    """that calculates the determinant of a matrix:
matrix: a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message
    matrix must be a square matrix
The list [[]] represents a 0x0 matrix
det = a*d - b*c
Returns: the determinant of matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0

    for col in range(len(matrix[0])):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det
