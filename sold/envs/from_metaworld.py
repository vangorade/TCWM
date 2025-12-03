import gymnasium as gym
import numpy as np
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.pixels import Pixels
from typing import Tuple


class GymnasiumToGymWrapper:
    """Wrapper to make Gymnasium envs compatible with old Gym API used in SOLD."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self, mode='rgb_array', **kwargs):
        if mode == 'rgb_array':
            return self.env.render()
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.reset(seed=seed)


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    """
    Create a MetaWorld environment using Gymnasium's native integration.
    
    Supported name formats:
    - 'ML1_pick-place-v3' -> Meta-World/ML1-pick-place-v3
    - 'ML10-train' -> Meta-World/ML10-train
    - 'ML45-train' -> Meta-World/ML45-train
    - 'ML45-test' -> Meta-World/ML45-test
    - 'MT50-train' -> Meta-World/MT50-train
    - Or direct Gymnasium ID: 'Meta-World/pick-place-v3'
    """
    # Parse the name to construct Gymnasium ID
    if name.startswith('Meta-World/'):
        gym_id = name
    elif name.startswith('ML') or name.startswith('MT'):
        # Handle ML1_task-name-v3 or ML10-train formats
        if '_' in name:
            benchmark, task = name.split('_', 1)
            gym_id = f'Meta-World/{benchmark}-{task}'
        else:
            gym_id = f'Meta-World/{name}'
    else:
        # Assume it's a task name, default to ML1
        gym_id = f'Meta-World/ML1-{name}'
    
    # Create the Gymnasium environment
    env = gym.make(gym_id, render_mode='rgb_array')
    env.reset(seed=seed)
    
    # Wrap to make compatible with old Gym API
    env = GymnasiumToGymWrapper(env)
    
    # Apply SOLD's standard wrappers
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    
    return env
