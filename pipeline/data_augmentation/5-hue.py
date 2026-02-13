#!/usr/bin/env python3
"""module for hue augmentation of an img for ML pipeline"""
import tensorflow as tf


def change_hue(image, delta):
    """that changes the hue of an image:
args:
    image: a 3D tf.Tensor containing the image to change
    delta: the amount the hue should change
Returns:
    the altered image"""
    img_hue = tf.image.adjust_hue(image, delta)
    return img_hue
