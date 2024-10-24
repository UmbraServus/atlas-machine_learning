#!/usr/bin/env python3
""" module for convolve"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding
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
Returns: a numpy.ndarray containing the convolved images"""
