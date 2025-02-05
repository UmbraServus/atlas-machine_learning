#!/usr/bin/env python3
import numpy as np
"""module for markov chain"""

def markov_chain(P, s, t=1):
    """
    method determining the probability of a Markov chain being in a particular
    state after a specified number of iterations.

    Parameters:
    P (numpy.ndarray): Square 2D array of shape (n, n) representing the
    transition matrix.
        P[i, j] is the probability of transitioning from state i to state j.
        n is number of states in the Markov chain.

    s (numpy.ndarray): Array of shape (1, n) representing the probability of
    starting in each state.
    t (int): Number of iterations that the Markov chain has been through.

    Returns: numpy.ndarray: Array of shape (1, n) representing the
    probability of being in a specific state after t iterations,
    or None on failure.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.ndim != 2 or s.ndim != 2:
        return None
    if P.shape[0] != P.shape[1] or P.shape[0] != s.shape[1]:
        return None
    if t < 1 or not isinstance(t, int):
        return None

    n = P.shape[0]
    current = s

    for _ in range(t):
        current = np.dot(current, P)

    return current