#!/usr/bin/env python3
"""Forecast BTC prices using LSTM on time series data."""
from preprocess import (
    init_wandb,
    load_time_series_data,
    preprocess_data,
    load_preprocessed_data
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# create window function
def create_window(data, window_size=1440, steps=60, batch_size=64, target_col='Close', features=None):
    """creates a sliding window of the data.
      Args:
        window_size: Number of timesteps to use as input (1440 = 24 hours for 1-min data)
        steps: How many steps ahead to predict (60 = 1 hour)
    """
    if features is None:
      raise ValueError('Features must be provided to get target_idx')

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + steps, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + steps))
    target_idx = features.index(target_col)
    dataset = dataset.map(lambda window: (window[:window_size], window[window_size + steps - 1, target_idx]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# create model
def create_rnn_model(window_size, num_features, lstm_units=128, num_layers=1, dropout_rate=0.2):
    """
    Create and compile an RNN/LSTM model for time series forecasting.

    Args:
        window_size (int): Number of time steps per input sequence
        num_features (int): Number of input features
        lstm_units (int): Number of units in the LSTM layer
        num_layers (int): Number of LSTM layers
        dropout_rate (float): Dropout rate between layers

    Returns:
        model (tf.keras.Model): Compiled Keras model
    """
    model = Sequential()

    # First LSTM layer
    if num_layers > 1:
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=(window_size, num_features)))
        model.add(Dropout(dropout_rate))

        # Middle LSTM layers
        for i in range(num_layers - 2):
            model.add(LSTM(lstm_units // (2 ** (i + 1)), return_sequences=True))
            model.add(Dropout(dropout_rate))

        # Last LSTM layer
        model.add(LSTM(lstm_units // (2 ** (num_layers - 1)), return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        # Single LSTM layer
        model.add(LSTM(lstm_units, input_shape=(window_size, num_features)))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# invert Scaling
def invert_scaling(pred_scaled, scaler_info, feature_name='Close'):
    """
    Convert scaled predictions back to real values.

    pred_scaled : np.array of shape (num_samples, ) or (num_samples, 1)
    scaler_info : dict with 'mean', 'std', 'features'
    feature_name: name of the column you want to invert
    """
    # find index of the feature
    idx = scaler_info['features'].index(feature_name)

    mean = scaler_info['mean'][idx]
    std = scaler_info['std'][idx]

    # invert
    pred_real = (pred_scaled * std) + mean
    return pred_real

# Main
def main():
    """Main function to orchestrate the workflow."""
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Configuration
    config = {
        'coinbase_path': '/content/drive/MyDrive/timeseries/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv (1).zip',
        'bitstamp_path': '/content/drive/MyDrive/timeseries/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv (1).zip',
        'preprocess_path': '/content/drive/MyDrive/timeseries/data/preprocess_data.npz',
        'window_size': 1440,
        'steps': 60,
        'batch_size': 4092,
        'lstm_units': 128,
        'num_layers': 1,
        'dropout_rate': 0.02,
        'epochs': 15,
        'learning_rate': 0.001,
        'force_preprocess': True,  # Set to True ONCE to regenerate with timestamps, then False
        'use_single_year': True,
        'year_to_use': 2014
    }

    # Initialize wandb
    wandb_config = init_wandb(config)

    # Load preprocessed data if available
    preprocessed = load_preprocessed_data(config['preprocess_path']) if not config['force_preprocess'] else None

    if preprocessed is not None and len(preprocessed) == 5:
        train_scaled, val_scaled, scaler_info, train_timestamps, val_timestamps = preprocessed
        print("Using cached preprocessed data with timestamps")
    else:
        # Load and preprocess data
        print("Loading data...")
        data = load_time_series_data(config['coinbase_path'], config['bitstamp_path'],
                                     explore=True, plot=False)

        # Filter single year if desired
        if config['use_single_year']:
            data = data[data.index.year == config['year_to_use']]
            print(f"📉 Using only data from year: {config['year_to_use']}")
            print(f"Filtered dataset size: {len(data)} rows")

        print("\nPreprocessing data...")
        train_scaled, val_scaled, scaler_info, train_timestamps, val_timestamps = preprocess_data(
            data, split=0.8, explore=True, save_path=config['preprocess_path']
        )

    # Create windowed datasets
    print("\nCreating windowed datasets...")
    train_dataset = create_window(train_scaled, config['window_size'], config['steps'],
                                  config['batch_size'], target_col='Close',
                                  features=scaler_info['features'])
    val_dataset = create_window(val_scaled, config['window_size'], config['steps'],
                                config['batch_size'], target_col='Close',
                                features=scaler_info['features'])

    # Create model
    print("\nCreating model...")
    num_features = len(scaler_info['features'])
    model = create_rnn_model(
        config['window_size'],
        num_features,
        config['lstm_units'],
        config['num_layers'],
        config['dropout_rate']
    )
    model.summary()

    # Log model architecture to wandb
    wandb.log({"model_parameters": model.count_params()})

    # Callbacks
    callbacks = [
        WandbMetricsLogger(log_freq='epoch'),
        WandbModelCheckpoint(
            filepath='/content/drive/MyDrive/timeseries/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # ------------------- PREDICT & PLOT -------------------
    print("\nPredicting on validation dataset...")

    # Get model predictions (scaled)
    pred_scaled = model.predict(val_dataset)

    # Extract actual targets (scaled)
    actual_scaled = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

    # Convert scaled values back to real BTC prices
    pred_real = invert_scaling(pred_scaled, scaler_info, feature_name='Close')
    actual_real = invert_scaling(actual_scaled, scaler_info, feature_name='Close')

    # Get corresponding timestamps
    window_size = config['window_size']
    steps = config['steps']
    num_predictions = len(actual_real)
    start_idx = window_size + steps - 1
    plot_timestamps = val_timestamps[start_idx:start_idx + num_predictions]

    # Debug information
    print(f"\n📊 Prediction Statistics:")
    print(f"Number of predictions: {num_predictions}")
    print(f"First timestamp: {pd.Timestamp(plot_timestamps[0])}")
    print(f"Last timestamp: {pd.Timestamp(plot_timestamps[-1])}")
    print(f"Time span: {pd.Timestamp(plot_timestamps[-1]) - pd.Timestamp(plot_timestamps[0])}")

    # Plot predictions vs actual WITH TIMESTAMPS
    plt.figure(figsize=(14, 6))
    plt.plot(plot_timestamps, actual_real, label='Actual BTC Close', alpha=0.8, linewidth=1.5)
    plt.plot(plot_timestamps, pred_real, label='Predicted BTC Close', alpha=0.8, linewidth=1.5)

    # Format x-axis for better date display
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Show every 10 days
    plt.gcf().autofmt_xdate()  # Rotate dates

    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.title('BTC Close Price Prediction (1-minute intervals)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/timeseries/prediction_vs_actual.png', dpi=150)
    print("Prediction plot saved.")
    plt.show()

    # ------------------- SAVE MODEL -------------------
    model.save('/content/drive/MyDrive/timeseries/final_model.keras')
    print("\nModel saved")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Model Training History - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Model Training History - MAE')

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/timeseries/training_history.png')
    print("Training history plot saved")
    plt.show()

    # Log final metrics to wandb
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    wandb.log({
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "overfitting_ratio": final_val_loss / final_train_loss
    })

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()