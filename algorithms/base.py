from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """Interface to standardize the training and evaluation of different algorithms.
    All the algorithms that are developped should inherith from the base class 
    such that they can be integrated in the training and eval loop. See 
    run_experiments.py"""
    
    @abstractmethod
    def reset(self):
        """Reset the internal state of the algorithm"""
        pass
    
    @abstractmethod
    def select_action(self, obs):
        """"Select an action based on observation"""
        pass

    @abstractmethod
    def train_step(self, transition):
        """Single training update step from the transition"""
        pass

    @abstractmethod
    def save(self, path):
        """Save the current state of the algorithm"""
        pass

    @abstractmethod
    def load(self, path):
        """Load a saved model"""
        pass

    
