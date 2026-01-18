#!/usr/bin/enc python3
"""module for having the agent play the environment"""
import numpy as np


def play(env, Q, max_steps=100):
    """method that has the trained agent play an episode:
args:
    env: the FrozenLakeEnv instance

    Q: numpy.ndarray containing the Q-table

    max_steps: the maximum number of steps in the episode

    You need to update 0-load_env.py to add render_mode="ansi"
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Ensure that the final state of the environment is also displayed
    after the episode concludes.

    Returns: The total rewards for the episode and
    a list of rendered outputs representing the board state at each step."""

    # Reset the environment to start a new episode
    state = env.reset()[0]

    total_rewards = 0
    rendered_outputs = []

    # Capture and store the initial state
    rendered_outputs.append(env.render())

    for step in range(max_steps):
        # Exploit: choose the action with the highest Q-value for current state
        action = np.argmax(Q[state, :])

        # Take the action
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Accumulate rewards
        total_rewards += reward

        # Capture and store the current state
        rendered_outputs.append(env.render())

        # Update state
        state = new_state

        # Break if episode is done
        if done:
            break

    return total_rewards, rendered_outputs
