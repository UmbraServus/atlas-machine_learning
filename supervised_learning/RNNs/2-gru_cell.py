#!/usr/bin/env python3
""" Create GRU cell class that represents a gated recurrent unit """
import numpy as np


class GRUCell():
    """Class GRUCell that represents a gated recurrent unit

    i (int): dimensionality of the data
    h (int): dimensionality of the hidden state
    o (int): dimensionality of the outputs

    Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
    that represent the weights and biases of the cell
    Wz and bz are for the update gate
    Wr and br are for the reset gate
    Wh and bh are for the intermediate hidden state
    Wy and by are for the output
    weights should be init using a rand norm distro in the order listed above
    The weights will be used on the right side for matrix multiplication
    The biases should be initialized as zeros"""

    def __init__(self, i, h, o):
        """Initialize the GRUCell"""
        self.Wz = np.random.normal(0, 1, (i + h, h))
        self.Wr = np.random.normal(0, 1, (i + h, h))
        self.Wh = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step

        x_t: np.ndarray of shape (m, i) that contains the data
        input for the cell
        m: the batch size for the data
        h_prev: a numpy.ndarray of shape (m, h) containing the
        prev hidden state

        The output of the cell should use a softmax activation function

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell"""

        # Concatenate previous hidden state and input data
        h_concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self.sigmoid(np.dot(h_concat, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.dot(h_concat, self.Wr) + self.br)

        # candidate hidden state
        candidate_h = (
            np.tanh(np.dot(np.concatenate((r * h_prev, x_t),
                                          axis=1), self.Wh) + self.bh)
        )
        # Next hidden state
        h_next = (1 - z) * h_prev + z * candidate_h

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
