#!/usr/bin/env python3
"""module for q-learning training method"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ method that performs Q-learning:
args:
    env: the FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    When the agent falls in a hole, the reward should be updated to be -1
    use epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

    Returns: Q, total_rewards
    Q: updated Q-table
    total_rewards: list containing the rewards per episode"""

    total_rewards = []
    # setup for loop to run thru each episode
    for episode in range(episodes):
        # exploitation vs exploration
        E = (min_epsilon +
             (epsilon - min_epsilon)*np.exp(-epsilon_decay*episode))
        state, _ = env.reset()
        episode_reward = 0
        # make sure it only goes a max n of steps
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, E)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            desc = env.unwrapped.desc.flatten()
            # -1 reward for falling in hole
            if terminated and desc[next_state] == b'H':
                reward = -1
            episode_reward += reward
            # update the Q table
            Q[state][action] = (Q[state][action] + alpha *
                                (reward + gamma * np.max(Q[next_state])
                                 - Q[state][action]))
            if done:
                break

            state = next_state
        # track total rewards
        total_rewards.append(episode_reward)
    return Q, total_rewards
