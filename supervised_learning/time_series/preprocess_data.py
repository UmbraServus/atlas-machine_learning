#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def load_time_series_data(file_path):
    """Load time series data from a CSV file."""
    data = pd.read_csv(file_path, index_col='Timestamp')
    data.index = pd.to_datetime(data.index, unit='s')
    data.sort_index(inplace=True)
    data = data.dropna(how='all')
    print("DataFrame Info:")
    data.info()
    print("\nDataFrame Head:")
    print(data.head())
    plot_col = ['Low', 'High', 'Volume_(BTC)', 'Volume_(Currency)']
    plot_feat = data[plot_col]
    plot_feat.index = data.index
    plot_feat.plot(subplots=True)
    plt.show()

def preprocess_data(data, target_column, window_size, split_ratio=0.8):
    """Preprocess the time series data for training and testing."""
    # Normalize the data
    data_mean = data.mean()
    data_std = data.std()
    norm_data = (data - data_mean) / data_std

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(norm_data) - window_size):
        sequences.append(norm_data.iloc[i:i + window_size].values)
        targets.append(norm_data.iloc[i + window_size][target_column])
    
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Split into training and testing sets
    split_index = int(len(sequences) * split_ratio)
    X_train, X_test = sequences[:split_index], sequences[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]

    return (X_train, y_train), (X_test, y_test), data_mean, data_std