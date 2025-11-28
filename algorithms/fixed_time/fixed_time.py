import numpy as np
from pathlib import Path
import json
import gymnasium as gym
from ..base import BaseAlgorithm


class FixedTimeAgent(BaseAlgorithm):
    """Fixed-time baseline that cycles through phases in a predetermined sequence.
    Uses the original SUMO traffic light program timing if not specified."""

    def __init__(
        self,
        env: gym.Env,
        cycle_phases: list[int] = None,
        phase_durations: list[int] = None,
    ):
        # Extract the CustomTrafficSignal object from the environment
        self.env = env
        self.ts = list(env.unwrapped.traffic_signals.values())[0]

        # Cycle through all green phases
        if cycle_phases is None:
            self.cycle_phases = list(range(self.ts.num_green_phases))
        else:
            self.cycle_phases = cycle_phases

        # Set the given phase durations or set equal phases when none set.
        if phase_durations is None:
            self.phase_durations = [1] * len(self.cycle_phases)
        else:
            self.phase_durations = phase_durations

        # Internal state
        self.current_phase_idx = 0
        self.steps_in_phase = 0

    def reset(self) -> None:
        """Reset to first phase in cycle."""
        self.current_phase_idx = 0
        self.steps_in_phase = 0

    def select_action(self, obs: np.ndarray, training:bool=True) -> int:
        """
        Select action based on fixed timing plan.
        Returns the current phase, then advances timer.
        """
        # Get current phase action
        action = self.cycle_phases[self.current_phase_idx]

        # Advance internal timer
        self.steps_in_phase += 1

        # Check if time to switch to next phase
        if self.steps_in_phase >= self.phase_durations[self.current_phase_idx]:
            self.current_phase_idx = (self.current_phase_idx + 1) % len(
                self.cycle_phases
            )
            self.steps_in_phase = 0

        return action

    def train_step(self, transition: tuple) -> None:
        pass

    def save(self, path: Path) -> None:
        """Save the fixed timing plan."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        config = {
            "cycle_phases": self.cycle_phases,
            "phase_durations": self.phase_durations,
        }

        with open(path, "w") as f:
            json.dump(config, f)

    def load(self, path: Path) -> None:
        """Load a fixed timing plan."""
        if Path(path).exists():
            with open(path, "r") as f:
                config = json.load(f)

            self.cycle_phases = config["cycle_phases"]
            self.phase_durations = config["phase_durations"]
            self.reset()
        else:
            print(f"Cannot load policy at path {path} because it does not exist.")
