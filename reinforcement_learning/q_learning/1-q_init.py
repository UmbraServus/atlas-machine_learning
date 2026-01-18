#!/usr/bin/env python3
"""module for initializing Q table"""
import numpy as np
import gymnasium as gym


def q_init(env):
    """that initializes the Q-table:

env: FrozenLakeEnv instance

Returns: the Q-table as a numpy.ndarray of zeros"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    return Q
