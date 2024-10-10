#!/usr/bin/env python3
""" module documentation """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ that updates the learning rate using inverse time decay in numpy:
        the learning rate decay should occur in a stepwise fashion
    args:
        alpha: the original learning rate
        decay_rate: weight used to determine the rate which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    Returns: the updated value for alpha """
