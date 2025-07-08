#!/usr/bin/env python3
""" module that performs forward propagation for a deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ performs forward propagation for a deep RNN:
    args:
    rnn_cells: list of RNNCell instances that will be used for the forward
    propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y"""
    
    # Get the number of time steps, batch size, and input dimensionality
    t, m, i = X.shape

    # Get the number of layers and the dimensionality of the hidden state
    layers = len(rnn_cells)
    h = rnn_cells[0].Wh.shape[1]

    # Initialize the hidden states and outputs
    H = np.zeros((layers + 1, t + 1, m, h))
    Y = np.zeros((layers, t, m, rnn_cells[0].Wy.shape[1]))
    H[0, 0] = h_0

    # Iterate through each time step
    for step in range(t):
        # Iterate through each layer
        for layer in range(layers):
            # Get the previous hidden state for the current layer
            h_prev = H[layer, step]
            # Get the input for the current layer
            x_t = X[step] if layer == 0 else H[layer - 1, step]
            # Perform forward propagation for the current layer
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            # Store the next hidden state and output
            H[layer + 1, step + 1] = h_next
            Y[layer, step] = y
        # Store the hidden states for the next time step
        H[0, step + 1] = h_next
    return H, Y
