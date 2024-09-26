#!/usr/bin/env python3
""" module doc """
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    """class for NeuralNetwork """

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
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    # getters
    @property
    def W1(self):
        """getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2

    # public methods

    def forward_prop(self, X):
        """ calculate the forward propagation of a Neural Network
            args:
                X: is an np array with shape (nx, m) contains input data
                    nx is # of input features
                    m is number of examples
                    """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.where(A2 >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
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
        error_dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(error_dZ2, A1.T)
        db2 = 1 / m * np.sum(error_dZ2, axis=1, keepdims=True)

        error_dZ1 = np.dot(self.__W2.T, error_dZ2) * (A1 * (1.0000001 - A1))
        dW1 = 1 / m * np.dot(error_dZ1, X.T)
        db1 = 1 / m * np.sum(error_dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

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
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if verbose and i % step == 0 or i == 0 or i == iterations:
                costs.append(cost)
                x_values.append(i)
                print(f"Cost after {i} iterations: {cost} every {step} iterations")

        if graph:
            plt.plot(x_values, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
