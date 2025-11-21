#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, GRU, Conv1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Imports Load/Explore/

# Initialize wandb at the top level
def init_wandb(config):
    """Initialize Weights & Biases with configuration."""
    wandb.init(
        project="btc-timeseries-forecast",  # Change this to your project name
        config=config,
        name=f"lstm_{config['lstm_units']}units_w{config['window_size']}_s{config['steps']}",
        tags=["lstm", "bitcoin", "time-series"]
    )
    return wandb.config

def load_time_series_data(coinbase_path, bitstamp_path, explore=True, plot=False):
    """Load time series data from CSV files."""
    # Read data
    print(f"Reading {coinbase_path}...")
    cb_data = pd.read_csv(coinbase_path, index_col='Timestamp')
    cb_data.index = pd.to_datetime(cb_data.index, unit='s')

    print(f"Reading {bitstamp_path}...")
    bs_data = pd.read_csv(bitstamp_path, index_col='Timestamp')
    bs_data.index = pd.to_datetime(bs_data.index, unit='s')

    data = pd.concat([cb_data, bs_data]).sort_index()
    data = data[~data.index.duplicated(keep="first")]

    if explore:
        data.info()
        print(data.head())
        print(data.describe().transpose())

        # Log data statistics to wandb
        wandb.log({
            "data_points": len(data),
            "date_range_days": (data.index[-1] - data.index[0]).days,
            "missing_values": data.isnull().sum().sum()
        })

        if plot:
            plot_col = ['Low', 'High', 'Volume_(BTC)', 'Volume_(Currency)', 'Open', 'Close']
            plot_feat = data[plot_col]
            plot_feat.index = data.index
            fig = plot_feat.plot(subplots=True, figsize=(12, 10))
            wandb.log({"data_overview": wandb.Image(plt)})
            plt.close()
    return data

def preprocess_data(data, split=0.8, explore=False, save_path='data/preprocess_data.npz'):
    """Preprocess time series data for training."""
    # Clean the data
    data['Volume_(BTC)'] = data['Volume_(BTC)'].fillna(0)
    data['Volume_(Currency)'] = data['Volume_(Currency)'].fillna(0)
    data['Weighted_Price'] = data['Weighted_Price'].ffill()
    data['Close'] = data['Close'].ffill()
    data['Open'] = data['Open'].ffill()
    data['High'] = data['High'].ffill()
    data['Low'] = data['Low'].ffill()

    if explore == True:
        data.info()
        print(data.head())
        print(data.describe().transpose())

    # Select features
    features = ['Weighted_Price',
                'Volume_(BTC)',
                'Volume_(Currency)',
                'Open',
                'Close',
                'High',
                'Low']

    # SAVE TIMESTAMPS BEFORE CONVERTING TO NUMPY
    timestamps = data.index.values
    data_values = data[features].values

    # Split train and val
    train_size = int(len(data_values) * split)
    train_data = data_values[:train_size]
    val_data = data_values[train_size:]

    train_timestamps = timestamps[:train_size]
    val_timestamps = timestamps[train_size:]

    # Normalize data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    train_scaled = (train_data - mean) / std
    val_scaled = (val_data - mean) / std

    scaler_info = {'mean': mean, 'std': std, 'features': features}

    # Save with timestamps
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        train=train_scaled,
        val=val_scaled,
        mean=scaler_info['mean'],
        std=scaler_info['std'],
        features=np.array(scaler_info['features'], dtype=object),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps
    )
    print(f"Preprocessed data saved to {save_path}")

    return train_scaled, val_scaled, scaler_info, train_timestamps, val_timestamps

# Load from saved file.
def load_preprocessed_data(load_path='data/preprocess_data.npz'):
    """Load preprocessed data from saved file."""
    if not os.path.exists(load_path):
        return None

    print(f"Loading preprocessed data from {load_path}...")
    loaded = np.load(load_path, allow_pickle=True)
    train_scaled = loaded['train']
    val_scaled = loaded['val']
    scaler_info = {
        'mean': loaded['mean'],
        'std': loaded['std'],
        'features': loaded['features'].tolist()
    }

    # Load timestamps if available
    train_timestamps = loaded.get('train_timestamps', None)
    val_timestamps = loaded.get('val_timestamps', None)

    return train_scaled, val_scaled, scaler_info, train_timestamps, val_timestamps