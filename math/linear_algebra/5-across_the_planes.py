#!/usr/bin/env python3
""" module for adding two arrays element wise """
def matrix_shape(matrix):
    """ function for matrix shape calc """
    shape = []
    matrix_stack = [matrix]

    while matrix_stack:
        current_stack = matrix_stack.pop()
        if isinstance(current_stack, list):
            shape.append(len(current_stack))
            if len(current_stack) > 0 and isinstance(current_stack[0], list):
                matrix_stack.append(current_stack[0])
    return shape


def add_matrices2D(mat1, mat2):
    """ method for adding matrices element wise. """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        result = []
        i = 0
        while i < len(mat1):
            x = 0
            arr3 = []
            arr1 = mat1[i]
            arr2 = mat2[i]
            while x < len(arr1):
                arr3.append(arr1[x] + arr2[x])
                x += 1
            result.append(arr3)
            i += 1
        return result
