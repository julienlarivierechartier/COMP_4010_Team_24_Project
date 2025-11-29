from ..base import BaseAlgorithm
from typing import Optional
from custom_env import CustomSumoEnvironment, DEAFULT_PED_WAIT_WEIGHT
from pathlib import Path
import numpy as np
import gymnasium as gym


class MaxPressureAgent(BaseAlgorithm):
    """BaseAlgorithm wrapper implementing MaxPressure control heuristic."""
    
    def __init__(self, num_obs:int, num_actions:int, ped_wait_weight: float = DEAFULT_PED_WAIT_WEIGHT):
        self.ped_wait_weight = ped_wait_weight
        self.ts = None # Don't grab it yet, wait for reset
    
    def reset(self):
        """
        CRITICAL FIX: logic to fetch the NEW TrafficSignal object.
        Sumo-RL destroys and recreates TS objects on every env.reset().
        We must update our reference here.
        """
        # We assume single agent for this specific project structure
        self.ts = list(self.env.unwrapped.traffic_signals.values())[0]
        
        # Apply the weight configuration to the new TS object
        self.ts.ped_wait_weight = self.ped_wait_weight
    
    def select_action(self, obs: np.ndarray, training:Optional[bool]=None):
        """MaxPressure queries SUMO directly via the stored TS object"""
        return self.ts.select_max_pressure_action()
    
    def train_step(self, transition: tuple):
        # No learning happens here
        pass
    
    def save(self, path: Path):
        pass
    
    def load(self, path: Path):
        pass

    def set_env(self, env:gym.Env):
        self.env = env