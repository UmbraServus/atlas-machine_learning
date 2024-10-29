#!/usr/bin/env python3
"""module for conv forward prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs fwd prop over a convolutional layer of a neural network
args:
    A_prev: np.ndarr shape (m, h_prev, w_prev, c_prev) output of prev layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    W: np.ndarr shape (kh, kw, c_prev, c_new) containing the kernels
        kh: the filter height
        kw: the filter width
        c_prev: number of channels in the previous layer
        c_new: number of channels in the output
    b: np.ndarr of shape (1, 1, 1, c_new) containing the biases
    activation: activation function applied to the convolution
    padding: string that is either same or valid
    stride: tuple of (sh, sw) containing the strides for the convolution
        sh: the stride for the height
        sw: the stride for the width
Returns: the output of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pad_w = ((w_prev - 1) * sh + kh - w_prev + 1) // 2

    elif padding == "valid":
        pad_h = 0
        pad_w = 0

    output_h = (h_prev + 2 * pad_h - kh) // sh + 1
    output_w = (w_prev + 2 * pad_w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_new))
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                        mode='constant')
    for h in range(output_h):
        for w in range(output_w):
            h_start = h * sh
            w_start = w * sw
            h_end = h_start + kh
            w_end = w_start + kw
            for k in range(c_new):
                current_slice = A_prev_pad[:, h_start:h_end,
                                           w_start:w_end, :]
                output[:, h, w, k] = np.sum(
                    current_slice * W[..., k], axis=(1, 2, 3)) + b[..., k]
    output = activation(output)
    return output
