#!/usr/bin/env python3
""" module for multiplying matrices together """


def mat_mul(mat1, mat2):
    """ method for multiplying matrices together """
    m, n = len(mat1), len(mat1[0])
    n2, p = len(mat2), len(mat2[0])

    if n != n2:
        return None
    else:
        mat3 = [[0 for _ in range(p)]for _ in range(m)]
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    mat3[i][j] += mat1[i][k] * mat2[k][j]
        return mat3
