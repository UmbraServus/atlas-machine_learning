#!/usr/bin/env python3
""" module for f1 score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ that calculates the F1 score of a confusion matrix
        F1 = 2 * (precision * sensitivity/recall) / (precision + recall)
    args:
        confusion: np.ndarr shape (classes, classes) row idxs represent the
            corr labels & col idxs repre the pred labels
    Returns: np.ndarray shape (classes,) containing F1 score of ea class """
    s = sensitivity(confusion)
    p = precision(confusion)
    F1 = 2 * (p * s) / (p + s)
    return F1