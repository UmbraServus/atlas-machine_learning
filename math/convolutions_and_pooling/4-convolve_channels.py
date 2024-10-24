#!/usr/bin/env python3
""" module for convolve"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ that performs a convolution on images with channels:
args:
    images: np.ndarr w/ shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernels: npy.ndarr w/ shape (kh, kw, c, nc) containin the kernels
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
    padding: either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0’s
    stride: a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
Returns: a numpy.ndarray containing the convolved images """
