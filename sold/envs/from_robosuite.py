import gym
import numpy as np
import torch
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.pixels import Pixels
from typing import Tuple
import robosuite as suite
from robosuite.wrappers import GymWrapper


class RobosuitePixelWrapper(gym.Wrapper):
    """Wrapper to extract pixel observations from Robosuite environments."""
    
    def __init__(self, env, camera_name='agentview', image_size=(64, 64), reward_scale=1000.0):
        super().__init__(env)
        self.camera_name = camera_name
        self.image_size = image_size
        self.reward_scale = reward_scale  # Scale rewards to make them more significant
        
        # Get reference to the underlying Robosuite environment
        self.robosuite_env = env
        while hasattr(self.robosuite_env, 'env'):
            self.robosuite_env = self.robosuite_env.env
        
        # Update observation space to be image-based
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(image_size[0], image_size[1], 3), 
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # GymWrapper returns (obs, info) tuple
        if isinstance(result, tuple):
            obs, info = result
            return self._get_pixel_obs(obs)
        else:
            return self._get_pixel_obs(result)
    
    def step(self, action):
        # Ensure action is a NumPy array for the underlying env
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        result = self.env.step(action)
        # GymWrapper returns 5 values (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # Scale reward to make it more significant for learning
        reward = reward * self.reward_scale
        
        # Extract pixel observation
        pixel_obs = self._get_pixel_obs(obs)
        
        # Ensure info is a dict
        if not isinstance(info, dict):
            info = {}
        
        # Get success information from the underlying Robosuite environment
        try:
            if hasattr(self.robosuite_env, '_check_success'):
                success = self.robosuite_env._check_success()
                info['success'] = bool(success)
                info['is_success'] = bool(success)
            elif hasattr(self.robosuite_env, 'check_success'):
                success = self.robosuite_env.check_success()
                info['success'] = bool(success)
                info['is_success'] = bool(success)
        except Exception:
            # If we can't get success info, set it to False
            info['success'] = False
            info['is_success'] = False
        
        return pixel_obs, reward, done, info
    
    def _get_pixel_obs(self, obs):
        """Extract pixel observation from the observation dict."""
        # GymWrapper with keys parameter returns the image directly as numpy array (flattened)
        if isinstance(obs, np.ndarray):
            # GymWrapper flattens the image, so reshape it back to (H, W, C)
            if len(obs.shape) == 1:
                # Reshape from flattened (H*W*C,) to (H, W, C)
                obs = obs.reshape(self.image_size[0], self.image_size[1], 3)
            image = obs
        elif isinstance(obs, dict):
            # Try to get the specified camera view
            if f'{self.camera_name}_image' in obs:
                image = obs[f'{self.camera_name}_image']
            elif self.camera_name in obs:
                image = obs[self.camera_name]
            else:
                # Fallback to first available image
                for key in obs.keys():
                    if 'image' in key.lower():
                        image = obs[key]
                        break
                else:
                    raise ValueError(f"No image observation found in obs dict. Available keys: {obs.keys()}")
        else:
            image = obs
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct shape (H, W, C)
        if len(image.shape) == 2:
            # Grayscale, add channel dimension
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # Channel first (C, H, W) -> transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            import cv2
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            # cv2.resize can drop the channel dimension if it's 1, so ensure it's preserved
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
        
        # Final shape check - ensure we have 3 channels
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Correct origin: MuJoCo offscreen returns images with bottom-left origin.
        # Flip vertically to standard top-left origin expected by PIL / viewers.
        image = np.flip(image, axis=0)
        
        return image.astype(np.uint8)


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0, reward_scale: float = 1000.0):
    """
    Create a Robosuite environment.
    
    Supported name formats:
    - 'Lift' -> Single-arm Lift task
    - 'Stack' -> Single-arm Stack task
    - 'PickPlace' -> Single-arm PickPlace task
    - 'NutAssembly' -> Single-arm NutAssembly task
    - 'Door' -> Single-arm Door task
    - 'Wipe' -> Single-arm Wipe task
    - Or any other Robosuite task name
    
    Optional suffixes:
    - Add '_Panda' or '_Sawyer' to specify robot (default: Panda)
    - Add '_frontview' or '_agentview' to specify camera (default: agentview)
    
    Examples:
    - 'Lift' -> Lift task with Panda robot, agentview camera
    - 'Stack_Sawyer' -> Stack task with Sawyer robot
    - 'PickPlace_frontview' -> PickPlace with frontview camera
    """
    # Parse the name to extract task, robot, and camera
    parts = name.split('_')
    task_name = parts[0]
    
    # Default settings
    robot = 'Panda'
    camera_name = 'agentview'
    
    # Parse optional suffixes
    for part in parts[1:]:
        if part in ['Panda', 'Sawyer', 'IIWA', 'Jaco', 'Kinova3', 'UR5e', 'Baxter']:
            robot = part
        elif part in ['agentview', 'frontview', 'birdview', 'sideview', 'robot0_eye_in_hand']:
            camera_name = part
    
    # Create Robosuite environment
    env = suite.make(
        env_name=task_name,
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_name,
        camera_heights=image_size[0],
        camera_widths=image_size[1],
        reward_shaping=True,  # Use dense rewards
        control_freq=20,  # Control frequency (Hz)
        horizon=max_episode_steps * action_repeat,  # Episode length in env steps
    )
    
    # Wrap with GymWrapper to make it compatible with Gym API
    env = GymWrapper(env, keys=[f'{camera_name}_image'])
    
    # Add pixel observation wrapper with reward scaling
    env = RobosuitePixelWrapper(env, camera_name=camera_name, image_size=image_size, reward_scale=reward_scale)
    
    # Apply SOLD's standard wrappers
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    
    # Seed the environment (Robosuite handles seeding internally during reset)
    # No need to call env.seed() as it's not supported by GymWrapper
    
    return env
