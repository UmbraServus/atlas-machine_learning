#!/usr/bin/env python3
""" This module contains the implementation of the forward algorithm for
a hidden Markov model. """
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
        - Transition[i, j] is the probability of transitioning from the hidden
          state i to j
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
      starting in a particular hidden state

    Returns:
    - P: the likelihood of the observations given the model
    - F: numpy.ndarray of shape (N, T) containing the forward path probablties
        - F[i, j] is the probability of being in hidden state i at time j givn
          the previous observations
    - None, None on failure"""

    # Initialize variables
    # Check for input validity
    # Initialize the forward path probability matrix F
    # Compute the initial probabilities
    # Iterate over each time step
        # Iterate over each state
            # Compute the forward probability for each state
    # Compute the likelihood of the observations
    # Return the likelihood and the forward path probability matrix
