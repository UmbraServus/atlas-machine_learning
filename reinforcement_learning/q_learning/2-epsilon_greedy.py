#!/usr/bin/env python3
""" module for epsilon greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """that uses epsilon-greedy to determine the next action:

Q: numpy.ndarray containing the q-table

state: the current state

epsilon: the epsilon to use for the calculation

You should sample p with numpy.random.uniformn to determine if your algorithm
should explore or exploit

If exploring, you should pick the next action with numpy.random.randint
from all possible actions

Returns: the next action index"""
    n_actions = Q.shape[1]
    if np.random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(0, n_actions)
    return action
