#!/usr/bin/env python3
"""module for randomly augmenting the brightness of an img."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """that randomly changes the brightness of an image:
args:
    image: a 3D tf.Tensor containing the image to change
    max_delta: the max amount the img should be brightened (or darkened)
Returns:
     the altered image"""

    altered_img = tf.image.random_brightness(image, max_delta)
    return altered_img