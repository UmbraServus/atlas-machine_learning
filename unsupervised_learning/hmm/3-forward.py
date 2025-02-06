#!/usr/bin/env python3
""" This module contains the implementation of the forward algorithm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Performs the forward algorithm for a hidden Markov model.
    Parameters:
    - Observation: numpy.ndarray of shape (T,) that contains the index of the
      observation
        - T is the number of observations
    - Emission: numpy.ndarray of shape (N, M) containing the emission
      probability of a specific observation given a hidden state
        - Emission[i, j] is the probability of observing j given the hidden
          state i
        - N is the number of hidden states
        - M is the number of all possible observations
    - Transition: 2D numpy.ndarray of shape (N, N) containing the transition
      probabilities
        - Transition[i, j] is the probability of transitioning from the hiddn
          state i to j
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
      starting in a particular hidden state

    Returns:
    - P: the likelihood of the observations given the model
    - F: numpy.ndarray of shape (N, T) containing the forward path probabltie
        - F[i, j] is the probability of being in hidden state i at time j gvn
          the previous observations
    - None, None on failure"""

    # Initialize variables
    N = Initial.shape[0]
    T = Observation.shape[0]
    # Check for input validity
    if len(Observation.shape) != 1:
        return None, None
    # Initialize the forward path probability matrix F
    F = np.zeros((N, T))
    # Compute the initial probabilities
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    # Iterate over each time step
    for t in range(1, T):
        # Iterate over each state
        for j in range(N):
            # Compute the forward probability for each state
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j] *
                             Emission[j, Observation[t]])
    # Compute the likelihood of the observations
    P = np.sum(F[:, T - 1])
    # Return the likelihood and the forward path probability matrix
    return P, F
