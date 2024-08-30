#!/usr/bin/env python3
""" module for calculating shape of matrix """

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
