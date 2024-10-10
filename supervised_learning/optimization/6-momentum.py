#!/usr/bin/env python3
"""module for setting up grad descent w/ momentum opti. algor. in tf"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """ that sets up the gradient descent with momentum optimization algorithm in TensorFlow:
    args:
        alpha is the learning rate.
        beta1 is the momentum weight.
    Returns: optimizer """

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
