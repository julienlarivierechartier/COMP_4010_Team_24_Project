import numpy as np
from pathlib import Path
from ..base import BaseAlgorithm
import gymnasium as gym

class RandomPolicy(BaseAlgorithm):
    """Random baseline that selects actions uniformly at random."""
    
    def __init__(self, env:gym.Env):
        self.env = env
    
    def reset(self) -> None:
        pass
    
    def select_action(self, obs: np.ndarray) -> int:
        """Select a random action."""
        return self.action_space.sample()
    
    def train_step(self, transition: tuple) -> None:
        pass
    
    def save(self, path: Path) -> None:
        pass
    
    def load(self, path: Path) -> None:
        pass