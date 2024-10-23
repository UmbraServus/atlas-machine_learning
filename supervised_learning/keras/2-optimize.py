#!/usr/bin/env python3
"""module for settin up adam optimization using keras """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization for keras model w/ categorical crossentropy
    loss and accuracy metrics
    args:
        network: the model to optimize
        alpha: learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter
    Returns: None"""
    model = network
    optimizer = K.optimizers.Adam(alpha, beta_1=beta1, beta_2=beta2)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
