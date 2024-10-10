#!/usr/bin/env python3
""" module documentation"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """ that sets up the Adam optimization algorithm in TensorFlow
    args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
    Returns: optimizer"""
