#!/usr/bin/env python3
""" deep neural network module """
import numpy as np


class DeepNeuralNetwork():
    """ DeepNeuralNetwork Class """

    def __init__(self, nx, layers):
        """ intialize nx and layers
            nx: number of input features
            layers: list representin the # of nodes in ea layer of the network
            """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                input_size = nx
            else:
                input_size = layers[i - 1]
            self.weights[f'W{i + 1}'] = (
                np.random.randn(layers[i], input_size)
                * np.sqrt(2. / input_size)
                )
            self.weights[f'b{i + 1}'] = np.zeros((layers[i], 1))
