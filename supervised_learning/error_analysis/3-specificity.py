#!/usr/bin/env python3
""" specificity module """
import numpy as np


def specificity(confusion):
    """ that calculates the specificity for each class in a confusion matrix
        Specificity = TN / FP + TN
        TN = Total#ofSamples - (FN + TP + FP)
    args:
        confusion: np.ndarr shape (classes, classes) row idxs represent the
            corr labels & col idxs repre the pred labels
            classes is the number of classes
    Returns: np.ndarr shape (classes,) containing specificity of ea class """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FN + TP + FP)
    specificity = TN / (FP + TN)
    return specificity
