#!/usr/bin/env python3
"""module for calculating accuracy of the model"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ that calculates the accuracy of a prediction
        args:
            y: a placeholder for the labels of the input data
            y_pred: a tensor containing the network’s predictions
        Returns: a tensor containing the decimal accuracy of the prediction
        """
    # get predictions
    predictions = tf.argmax(y_pred, 1)
    # get correct labels
    true_labels = tf.argmax(y, 1)
    # see which ones match up
    correct_predictions = tf.equal(predictions, true_labels)
    # cast everything to 1 or 0 float and get the mean of corr predictions
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
