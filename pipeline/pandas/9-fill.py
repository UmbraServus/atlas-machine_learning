#!/usr/bin/env python3
"""Module for filling using panda"""


def fill(df):
    """ Method takes a pd.DataFrame and Removes the Weighted_Price column.
Fills missing values in the Close column with the previous row’s value.
Fills missing values in the High, Low, and Open columns with the
corresponding Close value in the same row.
Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.

Returns:
    the modified pd.DataFrame."""

    # Drops weighted price
    df = df.drop(columns=['Weighted_Price'])

    # Fills missing values with previous row
    df['Close'] = df['Close'].ffill()

    # Fill missing values in High, Low, Open w/ corresponding Close.
    cols = ['High', 'Low', 'Open']
    for col in cols:
        df[col] = df[col].fillna(df['Close'])

    # Fills missing values with 0.
    cols = ['Volume_(BTC)', 'Volume_(Currency)']
    df[cols] = df[cols].fillna(0)

    return df
