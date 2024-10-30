#!/usr/bin/env python3
"""module for backward prop over pooling layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer of a neural network
args:
    dA: np.ndarr shape (m, h_new, w_new, c_new) containin the partial
    derivatives w/ respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev: np.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape: tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    stride: tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode: string containing either max or avg, indicating maximum or avg pool
Returns: the partial derivatives with respect to the previous layer (dA_prev)
"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    start_h = h * sh
                    start_w = w * sw
                    end_h = start_h + kh
                    end_w = start_w + kw
                    current_slice = A_prev[i, start_h:end_h, start_w:end_w, k]

                    if mode == 'max':
                        mask = (current_slice == np.max(current_slice))
                        dA_prev[i, start_h:end_h, start_w:end_w, k] += mask * dA[i, h, w, k]
                    elif mode == 'avg':
                        avg_dA = dA[i, h, w, k] / (kh * kw)
                        dA_prev[i, start_h:end_h, start_w:end_w, k] += np.ones((kh, kw)) * avg_dA
        return dA_prev
