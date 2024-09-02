#!/usr/bin/env python3
""" module for adding two arrays element wise """

def add_arrays(arr1, arr2):

    if len(arr1) != len(arr2):
        return None
    else:
        result = []
        i = 0
        while i < len(arr1):
            result.append(arr1[i] + arr2[i])
            i += 1

        return result
