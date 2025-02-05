#!/usr/bin/env python3
"""Module to determine if a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
    P (numpy.ndarray): A square 2D numpy.ndarray of shape (n, n) representing
                       the standard transition matrix. P[i, j] is the
                       probability of transitioning from state i to state j.
                       n is the number of states in the Markov chain.

    Returns:
    bool: True if the Markov chain is absorbing, or False on failure.
    """
    # Check if P is a square matrix
    if (not isinstance(P, np.ndarray) or
        P.ndim != 2 or
        P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]

    # Identify absorbing states (states with P[i, i] == 1)
    absorbing_states = np.diag(P) == 1

    if not np.any(absorbing_states):
        return False

    # Create a reachability matrix
    reachability = np.linalg.matrix_power(P, n)

    # Check if every state can reach an absorbing state
    for i in range(n):
        if not np.any(reachability[i, absorbing_states]):
            return False

    return True
