#!/usr/bin/env python3
""" module that defines the BiRNN class """
import numpy as np


class BidirectionalCell():
    """ class for a bidirectional RNN cell """
    def __init__(self, i, h, o):
        """
        class BidirectionalCell that represents a bidirectional RNN cell
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden states
        o (int): dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
        that represent the weights and biases of the cell
        Whf is for the hidden state in the forward direction
        Whb is for the hidden state in the backward direction
        Wy is for the outputs
        bhf is the bias for the hidden state in the forward direction
        bhb is the bias for the hidden state in the backward direction
        by is the bias for the outputs
        weights should be initialized using a random normal distribution
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
"""

        self.Whf = np.random.normal(0, 1, (i + h, h))
        self.Whb = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        h_prev: numpy.ndarray of shape (m, h) containing the previous
        hidden state in the forward direction
        x_t: numpy.ndarray of shape (m, i) that contains the data input
        for the cell
        m: the batch size for the data

        Returns: h_next
        h_next is the next hidden state in the forward direction
        """

        # Concatenate previous hidden state and input data
        h_concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state using tanh activation function
        h_next = np.tanh(np.dot(h_concat, self.Whf) + self.bhf)

        return h_next
