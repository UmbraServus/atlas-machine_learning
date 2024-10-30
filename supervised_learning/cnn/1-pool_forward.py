#!/usr/bin/env python3
"""module for fwd prop over a pooling layer"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs fwd prop over a pooling layer of a neural network
args:
    A_prev: np.ndarr shape (m, h_prev, w_prev, c_prev) output of prev layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    kernel_shape: tuple of (kh, kw) containing size of kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride: tuple of (sh, sw) containing the strides for the pooling
        sh: the stride for the height
        sw: the stride for the width
    mode: string containing either max or avg, indicating max or avg pooling
Returns: the output of the pooling layer """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_prev))
    for h in range(output_h):
        for w in range(output_w):
            start_h = h * sh
            start_w = w * sw
            end_h = start_h + kh
            end_w = start_w + kw
            current_slice = A_prev[:, start_h:end_h, start_w:end_w, :]
            if mode == 'max':
                output[:, h, w, :] = np.max(current_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, h, w, :] = np.mean(current_slice, axis=(1, 2))
    return output
