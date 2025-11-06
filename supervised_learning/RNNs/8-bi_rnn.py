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

    def backward(self, h_next, x_t):
        """
        performs backward propagation for one time step
        h_next: numpy.ndarray of shape (m, h) containing the next
        hidden state in the backward direction
        x_t: numpy.ndarray of shape (m, i) that contains the data input
        for the cell
        m: the batch size for the data

        Returns: h_prev
        h_prev is the previous hidden state in the backward direction
        """

        # Concatenate next hidden state and input data
        h_concat = np.concatenate((h_next, x_t), axis=1)

        # Compute the previous hidden state using tanh activation function
        h_prev = np.tanh(np.dot(h_concat, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """Softmax activation function with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def output(self, H):
        """ computes all outputs for the RNN
        H: numpy.ndarray of shape (t, m, 2h) that contains the
        concatenated hidden states from both directions,
        t is the number of time steps,
        m is the batch size,
        h is the dimensionality of the hidden states

        Returns: Y, numpy.ndarray of shape (t, m, o) that contains
        all the outputs"""
        # Get dimensions
        t, m, _ = H.shape
        o = self.Wy.shape[1]  # Output dimensionality

        # Initialize outputs
        Y = np.zeros((t, m, o))

        # Compute outputs for each time step
        for step in range(t):
            Y[step] = self.softmax(np.dot(H[step], self.Wy) + self.by)

        return Y

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
