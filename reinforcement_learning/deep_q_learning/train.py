#!/usr/bin/env python3
""" module to train an agent to play breakout from atari game system"""
# Import necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers.legacy import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class AtariProcessor(Processor):
    def process_observation(self, observation):
        return observation.astype(np.uint8)

    def process_state_batch(self, batch):
        batch = np.transpose(batch, (0, 2, 3, 1))
        return batch.astype("float32")/255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

# Wrapper to make Gymnasium compatible with keras-rl2
class GymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Check if the environment is an Atari emulator
        if hasattr(env.unwrapped, 'ale'):
            self.ale_env = env.unwrapped.ale
        else:
            # Gymnasium >=0.28 wraps Atari differently
            from ale_py import ALEInterface
            # Find the ALEInterface instance inside env.unwrapped
            if isinstance(env.unwrapped, gym.Env) and hasattr(env.unwrapped, 'ale'):
                self.ale_env = env.unwrapped.ale
            else:
                self.ale_env = None
        self.lives = 0

    def get_lives(self):
        """Safely return current number of lives"""
        if self.ale_env is not None:
            return self.ale_env.lives()
        else:
            return 0  # fallback if ALE not found

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.get_lives()

        # Auto-FIRE to start Breakout
        obs, reward, terminated, truncated, info = self.env.step(1)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        lives = self.get_lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives

        return obs, reward, done, info
# Create environment
env = gym.make('ALE/Breakout-v5', render_mode=None)
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                         frame_skip=4, noop_max=30)
env = GymnasiumWrapper(env)
# Get environment parameters
nb_actions = env.action_space.n
input_shape = (84,84, 4)

# Build the neural network model
model = Sequential()
model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())
# Configure the DQN agent
memory = SequentialMemory(limit=1000000, window_length=4)
# Base greedy policy
base_policy = EpsGreedyQPolicy()
# policy = base_policy
# Wrap in annealing
policy = LinearAnnealedPolicy(
    base_policy,        # the policy to anneal
    attr='eps',         # the attribute to change (epsilon)
    value_max=1.0,      # starting epsilon
    value_min=0.05,      # final epsilon
    value_test=0.05,    # epsilon when testing/evaluating
    nb_steps=600000)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=20000,
               target_model_update=10000,
               policy=policy,
               gamma=0.99,
               processor=AtariProcessor())
dqn.compile(RMSprop(learning_rate=0.00025, rho=.95, epsilon=0.01), metrics=['mae'])

# Train the agent
print("starting training...")
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

# Save the policy network
model.save('policy.h5', overwrite=True)

print("Training complete! Model saved to policy.h5")
env.close()
