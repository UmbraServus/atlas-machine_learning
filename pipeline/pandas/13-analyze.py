#!/usr/bin/env python3
"""module that computes statistics about the dataframe"""


def analyze(df):
    """method that takes a pd.DataFrame and:

Computes descriptive statistics for all columns except the Timestamp column.
Returns:
    new pd.DataFrame containing these statistics."""

    df = df.drop(columns= ['Timestamp']).describe()
    return df
