#!/usr/bin/env python3
"""module for making a df out of a dictionary"""
import pandas as pd

data = {
    'First': [0.0, 0.5, 1, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
