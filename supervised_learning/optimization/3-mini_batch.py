#!/usr/bin/env python3
""" module for creating minibatches """
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """ that creates mini-batches to be used for training a neural
    network using mini-batch gradient descent
    args:
        X is a numpy.ndarray of shape (m, nx) representing input data
            m is the number of data points
            nx is the number of features in X
        Y is a numpy.ndarray of shape (m, ny) representing the labels
            m is the same number of data points as in X
            ny is the number of classes for classification tasks.
        batch_size is the number of data points in a batch
    Returns: list of mini-batches containing tuples (X_batch, Y_batch)"""
    mini_batches = []
    m = X.shape[0]
    X_shuff, Y_shuff = shuffle_data(X, Y)
    for i in range(0, m, batch_size):
        end_idx = min(i + batch_size, m)
        X_batch = X_shuff[i:end_idx]
        Y_batch = Y_shuff[i:end_idx]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
