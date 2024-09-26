#!/usr/bin/env python3
""" module for one hot encoding """
import numpy as np


def one_hot_encode(Y, classes):
    """ method for one hot encoding - converts a numeric label
    vector into a one-hot matrix
        args:
            Y: np array w/ shape (m,) containing numerical class labels
                m: number of examples
            classes: maximum num of classes in Y
            """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 2:
        return None
    m = Y.shape[0]
    # initialze 2d array with zeros
    one_hot_encode = np.zeros((classes, m))
    # loop thru each label in Y
    for idx, label in enumerate(Y):
        # check to see if 0 <= label < classes(int)
        if label < classes:
            # encode it to 1 at the label row and idx column
            one_hot_encode[label, idx] = 1
        else:
            return None

    return one_hot_encode
