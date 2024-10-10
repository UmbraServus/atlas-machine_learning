#!/usr/bin/env python3
""" module documentaion """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ that sets up the RMSProp optimization algorithm in TensorFlow
    args:
        alpha: the learning rate
        beta2: the RMSProp weight (Discounting factor)
        epsilon: a small number to avoid division by zero
    Returns: optimizer """

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
