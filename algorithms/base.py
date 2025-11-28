from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseAlgorithm(ABC):
    """Interface to standardize the training and evaluation of different algorithms.
    All the algorithms that are developped should inherith from the base class
    such that they can be integrated in the training and eval loop. See
    run_experiments.py"""

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the algorithm"""
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action based on observation. For some algorithms this differs
        whether we are doing training or evaluation. Set the training flag
        accordingly"""
        pass

    @abstractmethod
    def train_step(self, transition: tuple) -> None:
        """Single training update step from the transition.
        transition = (state, action, reward, next_state, done)"""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the current state of the algorithm"""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a saved model"""
        pass
