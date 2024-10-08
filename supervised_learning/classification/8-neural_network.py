#!/usr/bin/env python3
""" module doc """
import numpy as np


class NeuralNetwork():
    """class for NeuralNetwork perfoming binary classification"""

    def __init__(self, nx, nodes):
        """ initialize nx and nodes
            nx: number of input featues
            nodes: number of nodes in the hidden layer
            """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0
