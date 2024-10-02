#!/usr/bin/env python3
""" module for function that returns two placeholders for x and y  using
    tensorflow. """
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ returns two placeholders, x and y, for the neural network:
        args:
            nx: the number of feature columns in our data
            classes: the number of classes in our classifier """
    x = tf.placeholder(tf.float32, shape=(None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classes))

    return x, y