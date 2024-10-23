#!/usr/bin/env python3
""" one hot module using keras """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ converts a label vector into a one-hot matrix
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix """
    one_hot_mat = K.utils.to_categorical(
        labels,
        num_classes=classes
    )
    return one_hot_mat
