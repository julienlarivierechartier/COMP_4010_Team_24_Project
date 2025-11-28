from ..base import BaseAlgorithm
from custom_env import CustomSumoEnvironment, DEAFULT_PED_WAIT_WEIGHT
from pathlib import Path
import numpy as np


class MaxPressureAgent(BaseAlgorithm):
    """BaseAlgorithm wrapper implementating MaxPressure control heuristic. Allows tuning
    the pedestrain wait weight"""

    def __init__(
        self,
        env: CustomSumoEnvironment,
        ped_wait_weight: float = DEAFULT_PED_WAIT_WEIGHT,
    ):
        self.env = env
        self.ts = list(env.unwrapped.traffic_signals.values())[0]
        # Set the weight for pedestrian waiting time in pressure calculation
        self.ts.ped_wait_weight = ped_wait_weight

    def reset(self):
        pass

    def select_action(self, obs: np.ndarray, training:bool=True):
        """MaxPressure queries SUMO directly and does not need observations"""
        return self.ts.select_max_pressure_action()

    def train_step(self, transition: np.ndarray):
        pass

    def save(self, path: Path):
        pass

    def load(self, path: Path):
        pass
