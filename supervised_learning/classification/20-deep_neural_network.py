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

    def cost(self, Y, A):
        """ calc the cost of the model using logi regression
            cost = - 1 / m * sum(Y * log(A) + (1 - Y) * log(1 - A)
            args:
                Y: np array w/ shape 1, m contains corr. labels for inp. data
                A: np array w/ shape 1, m containing activated output of neu-
                ron for each example
                """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ evals the network's predictions
            args:
                X: np array w/ shape nx, m conaints input data
                    nx number of input features
                    m number of examples
                Y: np array w/ shape 1, m that contains corr labels for input
                 data.
                 """
        A2, cache = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.where(A2 >= 0.5, 1, 0)
        return predictions, cost
