#!/usr/bin/env python3
"""module for slicing a df"""


def slice(df):
    """method that takes a pd.DataFrame and Extracts the columns High, Low,
    Close, and Volume_BTC.
        Selects every 60th row from these columns.

Returns:
    the sliced pd.DataFrame"""

    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df
