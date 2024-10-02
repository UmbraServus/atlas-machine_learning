#!/usr/bin/env python3
"""module for calculating accuracy of the model"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """that creates the training operation for the network:
    args:
        loss: the loss of the network’s prediction
        alpha: the learning rate
    Returns: an operation that trains the network using gradient descent"""

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
