#!/usr/bin/env python3
""" deep neural network module """
import numpy as np


class DeepNeuralNetwork():
    """ DeepNeuralNetwork Class """

    def __init__(self, nx, layers):
        """ intialize nx and layers
            nx: number of input features
            layers: list representin the # of nodes in ea layer of the network
            cache: A dictionary to hold all intermediary values of the network
                Upon instantiation, it should be set to an empty dictionary.
            """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = {}
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                input_size = nx
            else:
                input_size = layers[i - 1]
            self.__weights[f'W{i + 1}'] = (
                np.random.randn(layers[i], input_size)
                * np.sqrt(2. / input_size)
                )
            self.__weights[f'b{i + 1}'] = np.zeros((layers[i], 1))

        # getters
    @property
    def L(self):
        """getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """ getter for weights and bias"""
        return self.__weights

    # Public Methods

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network.
        args:
        X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights[f'W{i + 1}']
            b = self.__weights[f'b{i + 1}']
            A1 = self.cache[f'A{i}']
            z = np.dot(W, A1) + b
            A2 = 1 / (1 + np.exp(-z))
            self.__cache[f'A{i + 1}'] = A2
        return A2, self.__cache
