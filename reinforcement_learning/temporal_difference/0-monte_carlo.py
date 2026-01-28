#!/usr/bin/env python3
"""module for monte carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs Monte Carlo alg (episodic)

    args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing value estimates
        policy: function(state) -> action
        episodes: number of episodes to train
        max_steps: max steps per episode
        alpha: learning rate
        gamma: discount factor

    Returns:
        V: updated value estimates
    """

    for _ in range(episodes):
        episode = []

        # Reset environment
        state,_ = env.reset()
        state = int(state)

        # Generate a full episode
        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated

            episode.append((state, reward))
            state = int(next_state)

            if done:
                break

        # Monte Carlo return calculation (backwards)
        G = 0
        visited = set()
        for state, reward in reversed(episode):
            G = gamma * G + reward

            if state not in visited:
                V[state] = V[state] + alpha * (G - V[state])
                visited.add(state)

    return V
