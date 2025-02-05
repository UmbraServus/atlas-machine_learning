#!/usr/bin/env python3
"""module for regular markov chain"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular
    Markov chain.

    Parameters:
    P (numpy.ndarray): A square 2D numpy.ndarray of shape
                       (n, n) representing the transition
                       matrix. P[i, j] is the probability
                       of transitioning from state i to
                       state j. n is the number of states
                       in the Markov chain.
            n is the number of states in the Markov chain.
    Returns:
    numpy.ndarray: A numpy.ndarray of shape (1, n) containing
                   the steady state probabilities, or None
                   on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2 or \
       P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]
    # Check if the matrix is regular
    if not np.all(np.linalg.matrix_power(P, n) > 0):
        return None

    # Solve the equation πP = π with the constraint that the
    # sum of π is 1
    I = np.eye(n)
    A = np.append((P.T - I), np.ones((1, n)), axis=0)
    b = np.append(np.zeros(n), 1)
    steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
    return steady_state