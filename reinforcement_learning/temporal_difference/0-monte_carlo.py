#!/usr/bin/env python3
"""module for monte carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
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

    for episode in range(episodes):
        VisitedStates = []
        VisitedStatesRewards = []

        # Reset environment
        state, _ = env.reset()

        # Generate a full episode
        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated

            VisitedStatesRewards.append(int(reward))
            VisitedStates.append(int(state))
            state = next_state
            if done:
                break

        VisitedStates = np.array(VisitedStates)
        VisitedStatesRewards = np.array(VisitedStatesRewards)
        nStatesVisted = len(VisitedStates)

        # Monte Carlo return calculation (backwards)
        G = 0.0

        for episode_rev in reversed(range(nStatesVisted)):

            statetmp = VisitedStates[episode_rev]
            reward = VisitedStatesRewards[episode_rev]
            G = gamma * G + reward

            if statetmp not in VisitedStates[:episode]:
                V[statetmp] = V[statetmp] + alpha * (G - V[statetmp])

    return V
