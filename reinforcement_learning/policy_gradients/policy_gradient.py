#!/usr/bin/env python3
""" module for policy gradient """
import numpy as np


def policy(matrix, weight):
    """ Linear softmax policy for policy gradients.
    """
    # Ensure 1D feature vector representing states
    x = np.array(matrix).reshape(-1)

    # Linear logits
    action_logits = x.dot(weight)  # shape: (n_actions,)

    # Numerical stability trick (largest logit minus logits)
    action_logits -= np.max(action_logits)

    # Softmax
    exp_logits = np.exp(action_logits)
    probs = exp_logits / np.sum(exp_logits)

    return np.expand_dims(probs, axis=0)


def policy_gradient(matrix, weight):
    """ Policy gradient for policy gradients. using monte carlo method.

    args:
        matrix: matrix of features (representing states)
        weight: matrix of weights (representing actions)

    returns:
        action and the gradient of """

    probs = policy(matrix, weight)

    #sample an action from the probabilities.
    action = np.argmax(probs)

    # Compute gradient of log-policy
    probs = np.array(probs).reshape(-1) # 1d array for grad calc
    x = np.array(matrix).reshape(-1) # 1d array for grad calc
    grad = -probs[:, None] * x[None, :]   # shape: (n_actions, n_features)
    grad[action] = (1 - probs[action]) * x
    grad = grad.T # transpose so that the rows n column are flipped.

    return action, grad
