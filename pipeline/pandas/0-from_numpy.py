#!/usr/bin/env python3
"""module for creating a pandas df from a numpy array"""
import pandas as pd


def from_numpy(array):
    """method that creates a pd.DataFrame from a np.ndarray:
args:
    array: np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
Returns: the newly created pd.DataFrame"""
    n_col = array.shape[1]
    columns = [chr(65 + i) for i in range(n_col)]
    df = pd.DataFrame(array, columns=columns)
    return df
