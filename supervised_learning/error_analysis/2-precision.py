#!/usr/bin/env python3
""" precision module """
import numpy as np


def precision(confusion):
    """that calculates the precision for each class in a confusion matrix
        precision = TP / TP + FP
    args:
        confusion: np.ndarr shape (classes, classes) row idxs represent the
            corr labels & col idxs repre the pred labels
            classes is the number of classes
    Returns: np.ndarray shape (classes,) containing precision of ea class"""

    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    precision = TP / (TP + FP)
    return precision
