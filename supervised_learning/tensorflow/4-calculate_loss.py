#!/usr/bin/env python3
"""module for calculating accuracy of the model"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ that calculates the softmax cross-entropy loss of a prediction:
    args:
        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the networks predictions
    Returns: a tensor containing the loss of the prediction """
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )
    return loss
