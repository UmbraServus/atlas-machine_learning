#!/usr/bin/env python3
""" module for valid convolve"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """that performs a valid convolution on grayscale images
    args:
        images: np.ndarr w/ shape (m, h, w) containing multi gryscle imgs
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: np.ndarr w/ shape (kh, kw) containing the krnl 4 convolution
            kh is the height of the kernel
            kw is the width of the kernel
    Returns: np.ndarray containing the convolved images"""
