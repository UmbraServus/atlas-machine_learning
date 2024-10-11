#!/usr/bin/env python3
""" module documentation """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """ creates a learning rate decay operation in tensorflow
    using inverse time decay
        the learning rate decay should occur in a stepwise fashion
    args:
        alpha is the original learning rate
        decay_rate: w8 used to determine the rate at which alpha will decay
        decay_step: number of passes of grad descent that should occur before
        alpha is decayed further
    Returns: the learning rate decay operation """
    optimizer = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return optimizer