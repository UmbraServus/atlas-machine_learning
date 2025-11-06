#!/usr/bin/env python3
"""Create the class LSTMCell that represents a cell of an LSTM"""
import numpy as np


class LSTMCell():
    """ class LSTMCell that represents a cell of an LSTM"""

    def __init__(self, i, h, o):
        """Initialize the LSTMCell

        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs
        public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        represent the weights and biases of the cell

        Wfand bf are for the forget gate
        Wuand bu are for the update gate
        Wcand bc are for the candidate cell state
        Woand bo are for the output gate
        Wyand by are for the outputs
The weights should be init using a rand norm distro in the order listed above
The weights will be used on the right side for matrix multiplication
The biases should be initialized as zeros"""

        self.Wf = np.random.normal(0, 1, (i + h, h))
        self.Wu = np.random.normal(0, 1, (i + h, h))
        self.Wc = np.random.normal(0, 1, (i + h, h))
        self.Wo = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step

        x_t: np.ndarray of shape (m, i) that contains the data
        input for the cell
        m: the batch size for the data
        h_prev: a numpy.ndarray of shape (m, h) containing the
        previous hidden state
        c_prev: a numpy.ndarray of shape (m, h) containing the
        previous cell state

        The output of the cell should use a softmax activation function

        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell"""

        # Concatenate previous hidden state and input data
        h_concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        fg = self.sigmoid(np.dot(h_concat, self.Wf) + self.bf)

        # Update gate
        ug = self.sigmoid(np.dot(h_concat, self.Wu) + self.bu)

        # Candidate cell state
        c_hat = np.tanh(np.dot(h_concat, self.Wc) + self.bc)

        # Next cell state
        c_next = fg * c_prev + ug * c_hat

        # Output gate
        o = self.sigmoid(np.dot(h_concat, self.Wo) + self.bo)

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, c_next, y
