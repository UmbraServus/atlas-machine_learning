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
    layers, _, h =h_0.shape

    # Initialize the hidden states and outputs
    H = np.zeros((t + 1, layers, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))
    H[0] = h_0

    # Iterate through each time step
    for step in range(t):
        # Iterate through each layer
        for layer in range(layers):
            # Get the previous hidden state for the current layer
            h_prev = H[step, layer]
            # Get the input for the current layer
            x_t = X[step] if layer == 0 else H[step, layer - 1]
            # Perform forward propagation for the current layer
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            # Store the next hidden state and output
            H[step + 1, layer] = h_next
            if layer == layers - 1:
                # If it's the last layer, store the output
                Y[step] = y
    return H, Y
