#!/usr/bin/env python3
"""module for reversin row order & flipping the column & rows using pandas"""


def flip_switch(df):
    """that takes a pd.DataFrame and
    Sorts the data in reverse chronological order.
    Transposes the sorted dataframe.
Returns:
    the transformed pd.DataFrame."""

    df = df.sort_index(ascending=False).T

    return df
