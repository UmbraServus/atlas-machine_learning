#!/usr/bin/env python3
""" Module for performing the backward algorithm for a Hidden Markov Model
(HMM)."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model.

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
    - B: numpy.ndarray of shape (N, T) containing the backward path
      probabilities
        - B[i, j] is the probability of generating the future observations from
          hidden state i at time j
    - None, None on failure """

    # Pseudo code:
    # 1. Initialize variables:
    #    - T: number of observations
    #    - N: number of hidden states
    #    - B: a matrix of shape (N, T) to store backward probabilities
    # 2. Set the last column of B to 1 (since the probability of the future
    #    given the end is 1)
    # 3. Iterate backwards from the second last observation to the first:
    #    - For each state, calculate the backward probability using the
    #      transition and emission probabilities
    # 4. Calculate the likelihood of the observations given the model using
    #    the initial probabilities and the first column of B
    # 5. Return the likelihood and the backward matrix B
