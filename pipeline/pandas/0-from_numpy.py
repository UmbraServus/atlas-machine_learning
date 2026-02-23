#!/usr/bin/env python3
import pandas as pd
import numpy as np
import string


def from_numpy(array):
    """method that creates a pd.DataFrame from a np.ndarray:
args:
    array: np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
Returns: the newly created pd.DataFrame"""
    n_col = array.shape[1]
    columns = list(string.ascii_uppercase[:n_col])
    df = pd.DataFrame(array, columns=columns)
    return df