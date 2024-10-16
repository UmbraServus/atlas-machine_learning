#!/usr/bin/env python3
""" Confusion Matrix Module """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix
    args:
        labels: 1-hot np.ndarr shape (m, classes) containing correct labels
            m is the number of data points
            classes is the number of classes
        logits: 1-hot np.ndarr shape (m, classes) containing predicted labels
Returns: a confusion np.ndarray of shape (classes, classes) w/ row idxs
    representing true labels & col idxs representing the pred. labels"""

    classes = logits.shape[1]
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)
    confusion_mat = np.zeros((classes, classes), dtype=float)

    np.add.at(confusion_mat, (y_true, y_pred), 1)
    return confusion_mat
