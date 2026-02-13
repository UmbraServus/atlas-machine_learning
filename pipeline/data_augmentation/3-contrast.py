#!/usr/bin/env python3
"""module for augmenting the contrast of an image"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """method that randomly adjusts the contrast of an image.
args:
    image: 3D tf.Tensor reppin the input image to adjust the contrast.
    lower: float reppin the lower bound of the random contrast factor range.
    upper: float reppin the upper bound of the random contrast factor range.
Returns:
    contrast-adjusted image."""

    contrast_img = tf.image.random_contrast(image, lower=lower, upper=upper)
    return contrast_img
