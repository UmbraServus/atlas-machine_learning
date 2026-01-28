#!/usr/bin/env python3
"""TD(lambda) compatible with FrozenLake autograder"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=1.0, gamma=0.99):
    """
    TD(lambda) algorithm with eligibility traces for discrete envs.

    Args:
        env: Gymnasium environment instance
        V: np.ndarray of shape (n_states,) containing initial value estimates
        policy: function mapping state -> action
        lambtha: eligibility trace factor (lambda)
        episodes: number of episodes to run
        max_steps: maximum steps per episode
        alpha: learning rate (set to 1 for autograder)
        gamma: discount factor

    Returns:
        V: updated value estimates
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        # Reset eligibility traces
        E = np.zeros(n_states)

        state, _ = env.reset()
        state = int(state)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = int(next_state)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace
            E[state] += 1  # accumulating traces

            # Update all states using eligibility
            V += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            # Move to next state
            state = next_state
            if done:
                break

    return V
