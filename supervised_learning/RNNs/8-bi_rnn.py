#!/usr/bin/env python3
""" module that defines the bi rnn forward function """ 
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a Bidirectional RNN

    Args:
        bi_cell: instance of a BidirectionalCell with methods `forward`, `backward`
        X: input data of shape (t, m, i)
           t = timesteps, m = batch size, i = input size
        h_0: initial forward hidden state, shape (m, h)
        h_t: initial backward hidden state, shape (m, h)

    Returns:
        H: concatenated hidden states for all timesteps, shape (t, m, 2h)
        Y: outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = bi_cell.by.shape[1]

    # Containers for forward/backward hidden states
    H_fwd = np.zeros((t, m, h))
    H_bwd = np.zeros((t, m, h))

    # Forward direction (left → right)
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_fwd[step] = h_prev

    # Backward direction (right → left)
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_bwd[step] = h_next

    # Concatenate forward and backward hidden states (vectorized)
    H = np.concatenate((H_fwd, H_bwd), axis=2)  # shape: (t, m, 2h)

    # Vectorized output computation
    Y = np.dot(H, bi_cell.Wy) + bi_cell.by  # shape: (t, m, o)

    return H, Y
