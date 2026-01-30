#!usr/bin/env python3
""" module for policy gradient """
import numpy as np


def policy(matrix, weight):
    """ Linear softmax policy for policy gradients.
    """
    # Ensure 1D feature vector
    x = np.array(matrix).reshape(-1)

    # Linear logits
    action_logits =  x.dot(weight)  # shape: (n_actions,)

    # Numerical stability trick (largest logit minus logits)
    action_logits -= np.max(action_logits)

    # Softmax
    exp_logits = np.exp(action_logits)
    probs = exp_logits / np.sum(exp_logits)

    return probs
