#!/usr/bin/env python3
"""module for forward prop of the neural network"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]): 
    """that creates the forward propagation graph for the neural network
    args:
        x: the placeholder for the input data
        layer_sizes: a list containing the number of nodes in each layer of
        the network
            layer_output first x is the first input in the network, then
            the rest are the subsequent layers outputs.
        activations: a list containing the activation functions for each
        layer of the network
    Returns: the prediction of the network in tensor form """

    layer_output = x

    for i in range(len(layer_sizes)):
        layer_output = create_layer(
            layer_output,
            layer_sizes[i], 
            activations[i]
            )
    return layer_output
