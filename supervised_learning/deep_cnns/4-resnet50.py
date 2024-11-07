#!/usr/bin/env python3
"""module that builds a resnet50"""
from tensorflow import keras as K


def resnet50():
    """that builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015):

You can assume the input data will have shape (224, 224, 3)

All convolutions inside and outside the blocks should be followed by
batch normalization along the channels axis and
a rectified linear activation (ReLU), respectively.

All weights should use he normal initialization
The seed for the he_normal initializer should be set to zero
You may use:
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block

Returns: the keras model"""
