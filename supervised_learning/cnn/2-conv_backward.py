#!/usr/bin/env python3
"""module for conv backward prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back prop. over a convolutional layer of a neural network
args:
    dZ: np.ndarr shape (m, h_new, w_new, c_new) containin the partial
        derivatives w/ respect to the unactvted output of the conv layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev: np.ndarr shape (m, h_prev, w_prev, c_prev) containin the output
        of the prev layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W: np.ndarr shape (kh, kw, c_prev, c_new) containin the ks for the conv.
        kh is the filter height
        kw is the filter width
    b: np.ndarr shape (1, 1, 1, c_new) containin the biases applied to conv
    padding: string that is either same or valid, indicating type of padding
    stride: tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
Returns: partial derivatives w/ respect to the previous layer (dA_prev), the
    kernels (dW), and the biases (db), respectively"""

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    kh, kw, _, _ = W.shape

    if padding == 'same':
        pad_h = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pad_w = ((w_prev - 1) * sw + kh - w_prev + 1) // 2

    if padding == 'valid':
        pad_h = 0
        pad_w = 0

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                    (pad_w, pad_w), (0, 0)), mode='constant')
    dA_prev = np.zeros_like(A_prev_pad)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                h_start = h * sh
                w_start = w * sw
                h_end = h_start + kh
                w_end = w_start + kw
                for k in range(c_new):
                    current_slice = A_prev_pad[i,
                                               h_start:h_end,
                                               w_start:w_end,
                                               :]
                    dW[:, :, :, k] += current_slice * dZ[i, h, w, k]
                    dA_prev[i,
                            h_start:h_end,
                            w_start:w_end,
                            :] += W[:, :, :, k] * dZ[i, h, w, k]

    if padding == 'same':
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    elif padding == 'valid':
        dA_prev = dA_prev[:, :h_prev, :w_prev, :]

    return dA_prev, dW, db
