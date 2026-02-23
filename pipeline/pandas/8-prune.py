#!/usr/bin/env python3
"""Module for pruning NaN values"""


def prune(df):
    """Prune where Close has NaN values"""
    return df.dropna(subset=['Close'])