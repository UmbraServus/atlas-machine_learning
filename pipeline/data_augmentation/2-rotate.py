#!/usr/bin/env python3
"""module for rotating an img 90 degrees"""
import tensorflow as tf


def rotate_image(image):
    """that rotates an image by 90 degrees counter-clockwise:
arg:
    image: a 3D tf.Tensor containing the image to rotate

Returns:
     rotated image"""

    rotated_image = tf.image.rot90(image)
    return rotated_image
