#!/usr/bin/env python3
"""module that takes a df and returns a numpy array"""


def array(df):
    """method that takes a pd.DataFrame as input and performs the following
args:
    df: a pd.DataFrame containing columns named High and Close.
The function should select the last 10 rows of the High and Close columns.
Convert these selected values into a numpy.ndarray.
Returns: the numpy.ndarray"""
    arr = df[['High', 'Close']].tail(10).to_numpy()
    return arr
