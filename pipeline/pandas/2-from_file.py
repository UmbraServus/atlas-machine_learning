#!/usr/bin/env python3
"""module for loading data from a file as a dataframe"""
import pandas as pd


def from_file(filename, delimiter):
    """method that loads data from a file as a pd.DataFrame:
args:
    filename is the file to load from
    delimiter is the column separator
Returns:
    the loaded pd.DataFrame"""

    df = pd.read_csv(filename, sep=delimiter)
    return df
