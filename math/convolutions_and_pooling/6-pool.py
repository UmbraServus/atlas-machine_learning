#!/usr/bin/env python3
""" module for pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images
    args:
        images: np.ndarr w/ shape (m, h, w, c) containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape: tuple of (kh, kw) containing the kernel shape 4 pooling
        kh: the height of the kernel
        kw: the width of the kernel
        stride: a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode: type of pooling
            max indicates max pooling
            avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images"""
