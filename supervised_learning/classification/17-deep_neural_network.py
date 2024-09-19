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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l in range(self.__L):
            if not isinstance(layers[l], int) or layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            if l == 0:
                input_size = nx
            else:
                input_size = layers[l - 1]
            self.__weights[f'W{l + 1}'] = (np.random.randn(layers[l], input_size)
                                           * np.sqrt(2. / input_size))
            self.__weights[f'b{l + 1}'] = np.zeros((layers[l], 1))

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
