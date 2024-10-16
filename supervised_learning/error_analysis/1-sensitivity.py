#!/usr/bin/env python3
""" sensitivity module """
import numpy as np


def sensitivity(confusion):
    """that calculates the sensitivity for each class in a confusion matrix
        Sensitivity = TP / TP + FN
    args:
        confusion: np.ndarr shape (classes, classes) row idxs represent the
            corr labels & col idxs repre the pred labels
            classes is the number of classes
    Returns: np.ndarray shape (classes,) containing sensitivity of ea class"""
    sensitivity = np.zeros(len(confusion))
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    sensitivity = TP / (FN + TP)
    return sensitivity
