#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file
fill = __import__('9-fill').fill

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = fill(df)

# Rename the col Timestamp to Date & Conv the timestamp values to date values
df = df.rename(columns= {"Timestamp":"Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')

# Index the data frame on Date
df = df.set_index("Date")

df = df['2017':]
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum',
})
print(df)
df.plot()
plt.show()
