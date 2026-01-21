#!/usr/bin/env python3
""" module to train an agent to play breakout from atari game system"""
# Import necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

class AtariProcessor(Processor):
    def process_observation(self, observation):
        return observation.astype(np.uint8)

    def process_state_batch(self, batch):
        batch = np.transpose(batch, (0, 2, 3, 1))
        return batch.astype("float32")/255.0

# Wrapper to make Gymnasium compatible with keras-rl2
class GymnasiumWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render()

# Create environment
env = gym.make('ALE/Breakout-v5', render_mode="human")
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                         frame_skip=1, noop_max=30)
env = GymnasiumWrapper(env)

# Get environment parameters
nb_actions = env.action_space.n
input_shape = (84, 84, 4)

# load the neural network weights


# Build the neural network model and load weights from trained model.
model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

trained_model = load_model('policy.h5')
model.set_weights(trained_model.get_weights())

# Configure the DQN agent
memory = SequentialMemory(limit=1000000, window_length=4)
policy = EpsGreedyQPolicy(eps=0.05)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=0,
               target_model_update=10000,
               policy=policy,
               processor=AtariProcessor())
dqn.compile(Adam(lr=0.00025), metrics=['mae'])

print("starting game visualization...")

# Set nb_episodes to 5 or 10
# Set visualize=True to see the agent play the game in real-time
dqn.test(env=env, nb_episodes=10, visualize=True, verbose=2)
env.close()

