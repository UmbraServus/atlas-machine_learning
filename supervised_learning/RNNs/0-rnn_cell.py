#!/usr/bin/env python3
import numpy as np
"""Create the class RNNCell that represents a cell of a simple RNN"""


class RNNCell():
    """Class RNNCell that represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
    Initialize the RNNCell:

    i (int): dimensionality of the data
    h (int): dimensionality of the hidden state
    o (int): dimensionality of the outputs

    public instance attributes Wh, Wy, bh, by that represent the weights
    and biases of the cell
    Wh and bh are for the concatenated hidden state and input data
    Wy and by are for the output
    weights should be init usin a rand norm distro in the order listed above
    The weights will be used on the right side for matrix multiplication
    The biases should be initialized as zeros"""
        self.Wh = np.random.normal(0, 1, (h + i, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step
    args:
    x_t: np.ndarray of shape (m, i) that contains the data input for the cell
    m: the batch size for the data
    h_prev: a numpy.ndarray of shape (m, h) containing the prev hidden state

    The output of the cell should use a softmax activation function

    Returns: h_next, y
    h_next is the next hidden state
    y is the output of the cell"""
        # concat the previous hidden state and the input data
        h_concat = np.concatenate((h_prev, x_t), axis=1)
        # calculate the next hidden state and output
        h_next = np.tanh(np.dot(h_concat, self.Wh) + self.bh)
        # multiply the next hidden state with Wy and add by to get the output
        output = np.dot(h_next, self.Wy) + self.by
        # apply softmax activation function to the output
        output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

        return h_next, output
