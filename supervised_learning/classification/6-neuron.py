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
            cost = - 1 / m * sum(Y * log(A) + (1 - Y) * log(1 - A)
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
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron
        and updates the private b and W attributes.
        Args:
            X: "numpy.ndarray w/ shape" (nx, m) tht contains input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: " " (1, m) tht contains correct labels for the input data
            A: " " (1, m) containin the activ8d outp. of the neuron for ea ex
            """
        m = Y.shape[1]
        error = A - Y
        dW = 1 / m * np.dot(error, X.T)
        db = 1 / m * np.sum(error)
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron
            args:
                X: ndarray shape (nx, m) tht contains the input data
                    nx is the number of input features to the neuron
                    m is the number of examples
                Y: ndarray shape (1, m) tht contns corrct lbls for input data
                iterations: the number of iterations to train over
                alpha is the learning rate """
        # exceptions
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        
        # loop over iterations
        for i in range(iterations):

        # forward prop
            A = self.forward_prop(X)

        # calc gradients
            self.gradient_descent(X, Y, A, alpha)

        # return evaluation
        return self.evaluate(X, Y)
