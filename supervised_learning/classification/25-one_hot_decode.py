#!/usr/bin/env python3
""" module doc """
import numpy as np


def one_hot_decode(one_hot):
    """ one hot decode converts a one hot matrix into a vector of labels
      args:
        one_hot: one-hot encoded ndarray with shape (classes, m)
            classes: the maximum number of classes
            m: the number of examples"""
    if not isinstance(one_hot, np.ndarray) or one_hot.shape[0] < 2:
        return None
    return np.argmax(one_hot, axis=0)
