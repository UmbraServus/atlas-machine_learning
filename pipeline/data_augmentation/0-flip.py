#!/usr/bin/env python3
"""module for flipping an image"""
import tensorflow as tf


def flip_image(image):
  """ method that flips an image horizontally:
args:
    image: a 3D tf.Tensor containing the image to flip
Returns:
    the flipped image"""

  flipped_image = tf.image.flip_left_right(image)

  return flipped_image
