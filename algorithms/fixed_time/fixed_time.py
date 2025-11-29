import numpy as np
from pathlib import Path
import json
import gymnasium as gym
from ..base import BaseAlgorithm


class FixedTimeAgent(BaseAlgorithm):
    """Fixed-time baseline that cycles through phases in a predetermined sequence.
    Uses the original SUMO traffic light program timing."""

    def __init__(
        self,
        env: gym.Env,
    ):
        # Reference to the SUMO environment
        self.env = env
        # Extract the first traffic signal object
        self.ts = list(env.unwrapped.traffic_signals.values())[0]
        
        # Fixed cycle of green phases
        self.cycle_phases = list(range(self.ts.num_green_phases))
        self.current_phase_idx = 0
        self.steps_in_phase = 0

    def reset(self) -> None:
        """Reset to first phase in cycle."""
        self.current_phase_idx = 0
        self.steps_in_phase = 0

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        action = self.cycle_phases[self.current_phase_idx]

        self.steps_in_phase += 1

        # Use the underlying SUMO-RL environment's delta_time
        delta_time = self.env.unwrapped.delta_time
        phase_duration_steps = int(self.ts.all_phases[action].duration / delta_time)

        if self.steps_in_phase >= phase_duration_steps:
            self.current_phase_idx = (self.current_phase_idx + 1) % len(self.cycle_phases)
            next_phase = self.cycle_phases[self.current_phase_idx]
            self.ts.set_next_phase(next_phase)
            self.steps_in_phase = 0

        return action


    def train_step(self, transition: tuple) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass