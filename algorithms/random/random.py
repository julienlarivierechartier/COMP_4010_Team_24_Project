from ..base import BaseAlgorithm
import gymnasium as gym
from pathlib import Path
import numpy as np


class RandomAgent(BaseAlgorithm):
    """Random baseline that selects actions uniformly at random."""

    def __init__(self, env: gym.Env):
        self.env = env

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray, training:bool=True) -> int:
        """Select a random action."""
        return self.env.action_space.sample()

    def train_step(self, transition: tuple) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
