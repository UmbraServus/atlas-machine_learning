#!/usr/bin/env python3
""" Training function for policy gradient """
import numpy as np


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ Implements full training using policy gradient.

    args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, render the environment every 1000 episodes

    returns:
        all values of the score (sum of all rewards during one episode)
    """
    policy_gradient = __import__('policy_gradient').policy_gradient

    # Get state and action dimensions from environment
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize weight matrix
    weight = np.random.rand(n_features, n_actions)

    # Store all scores
    scores = []

    for episode in range(nb_episodes):
        # Reset environment
        state, _ = env.reset()

        # Storage for episode trajectory
        gradients = []
        rewards = []

        done = False
        truncated = False

        # Determine if we should render this episode
        render = show_result and (episode % 1000 == 0)

        # Collect episode trajectory
        while not (done or truncated):
            # Render if needed
            if render:
                env.render()

            # Get action and gradient
            action, grad = policy_gradient(state, weight)

            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Store trajectory
            rewards.append(reward)
            gradients.append(grad)

            state = next_state

        # Calculate score (total reward for episode)
        score = sum(rewards)
        scores.append(score)

        # Calculate discounted returns (G_t) for each timestep
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Normalize returns (baseline)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

        # Update weights using policy gradient
        for t in range(len(gradients)):
            weight += alpha * returns[t] * gradients[t]

        # Print episode info
        print(f"Episode: {episode} Score: {score}")

    return scores
