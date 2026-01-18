#!/usr/bin/env python3
import gymnasium as gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """that loads the pre-made FrozenLakeEnv evnironment from gymnasium:

desc: None or a list of lists containing a custom description of the map
to load for the environment

map_name: None or a string containing the pre-made map to load
Note: If both desc and map_name are None, the environment will load a
randomly generated 8x8 map

is_slippery: boolean to determine if the ice is slippery

Returns: environment"""
    if desc is None and map_name is None:
        map_name = "8x8"

    environment = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=is_slippery)

    return environment
