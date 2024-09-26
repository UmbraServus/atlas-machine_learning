#!/usr/bin/env python3
""" deep neural network module """
import numpy as np
import matplotlib.pyplot as plt

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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
            and updates private b and W attributes.
            args:
                X: a numpy.ndarray with shape (nx, m) contains input data
                    nx: the number of input features to the neuron
                    m: the number of examples
                Y: np array w/ shape 1, m that contains the corr. labels for
                the input data
                A1: the output of the hidden layer
                A2: the predicted output
                alpha: the learning rate"""
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            prev_A = cache[f"A{i-1}"]
            dW = 1 / m * np.dot(dZ, prev_A.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

            dZ = (
                np.dot(self.__weights[f"W{i}"].T, dZ)
                * (prev_A * (1 - prev_A))
                  )
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        x_values = []
        for i in range(iterations):
            A2, cache = self.forward_prop(X)
            cost = self.cost(Y, A2)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0 or i == 0 or i == iterations:
                costs.append(cost)
                x_values.append(i)
                print(f"Cost after {i} iterations: {cost}")        
        if graph:
            plt.plot(x_values, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)


