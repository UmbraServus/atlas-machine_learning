#!/usr/bin/env python3
"""Module for pruning NaN values"""
import pandas as pd


def prune(df):
    """Prune where Close has NaN values"""
    return df.dropna(subset=['Close'])