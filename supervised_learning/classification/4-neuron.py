#!/usr/bin/env python3
"""module for single neuron performing binary classification"""
import numpy as np


class Neuron():
    """ class for a single Neuron performing binary classification """

    def __init__(self, nx):
        """ initialize neuron
            args:
                nx: number of input features """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

# getters

    @property
    def W(self):
        """ gettter for W"""
        return self.__W

    @property
    def b(self):
        """ getter for b """
        return self.__b

    @property
    def A(self):
        """ getter for A """
        return self.__A

# public methods

    def forward_prop(self, X):
        """ method for forward prop. in a binary classification
        args:
            X: numpy.ndarray w/ shape (nx, m) that conatains input data
                m is the number of examples
                nx is the number of  input features to the neuron """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
            cost = - 1 / m * sum(Y * log(__A) + (1 - Y) * log(1 - __A)
            args:
                Y: "numpy.ndarray w shape (1, m)" contains labels for input
                A: "  " contains the activated output of the neuron for ea ex.
                    m is the number of examples.
                """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) +
                               (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ method to evaluate the neurons predictions. meaning returning
        cost and prediction based on activation output and comparison to
        actual label data.
            args:
                X: "numpy.ndarray w/ shape" (nx, m), contains the input data
                    nx is the number of input features to the neuron
                    m is the number of examples
                Y: " " (1, m), contains the correct labels for the input data
                """
        predictions = self.forward_prop(X)
        predictions = (self.__A >= 0.5).astype(int)
        cost = self.cost(Y, self.__A)
        return predictions, cost
