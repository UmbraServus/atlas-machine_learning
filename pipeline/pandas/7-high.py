#!/usr/bin/env python3
"""module for sorting in descending order"""


def high(df):
    """that takes a pd.DataFrame and
    Sorts it by the High price in descending order.
Returns:
    the sorted pd.DataFrame."""
    return df.sort_values(by=['High'], ascending=False)
