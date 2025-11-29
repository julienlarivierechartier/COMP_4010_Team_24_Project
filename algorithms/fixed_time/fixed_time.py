import numpy as np
from pathlib import Path
import gymnasium as gym
from ..base import BaseAlgorithm

class FixedTimeAgent(BaseAlgorithm):
    """This class ensures that a fixed time agent conforms to the BaseAlgorithm class.
    In SUMO-RL, the fixed time policy is set when creating the environment hence there
    is no need control via an agent."""
    
    def __init__(self, num_obs:int, num_actions:int):
        pass

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Fixed time ignores the action given to env.step() so return any action"""
        return 0

    def train_step(self, transition: tuple) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
    
    def set_env(self, env:gym.Env):
        pass