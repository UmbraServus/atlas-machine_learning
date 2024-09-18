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
