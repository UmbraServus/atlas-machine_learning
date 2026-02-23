#!/usr/bin/env python3
"""module for turning timestamp from column to index in a dataframe"""
import pandas as pd


def index(df):
    """method takes a pd.DataFrame and Sets the Timestamp column as
    the index of the dataframe.
Returns:
    the modified pd.DataFrame."""

    df = df.set_index('Timestamp')
    return df
