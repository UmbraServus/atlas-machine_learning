#!/usr/bin/env python3
"""module for cropping an image"""
import tensorflow as tf


def crop_image(image, size):
    """ method that performs a random crop of an image:
args:
    image: a 3D tf.Tensor containing the image to crop
    size: a tuple containing the size of the crop
Returns:
    the cropped image"""

    cropped_img = tf.image.random_crop(image, size)

    return cropped_img
