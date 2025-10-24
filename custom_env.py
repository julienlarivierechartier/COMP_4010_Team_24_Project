"""-------------------------------------------------------------------------------------
File: custom_env.py
Description: Implementation of an ObservationFunction class and a rewards function for
the Env Demo project deliverable. Importing this file registers the custom environment
to the Gymnasium API. The NET_FILE_PATH and ROUTE_FILE_PATH have to be designed for this
to work.
-------------------------------------------------------------------------------------"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from sumo_rl.environment.env import SumoEnvironment, TrafficSignal, ObservationFunction

# Id of the custom environment registered to Gymnasium API
CUSTOM_ENV_ID = "custom-tsc-env-v0"

"""
The traffic scenario map. Need to create those 2 files using the `net` GUI tool in 
${SUMO_HOME}/tools)"""
NET_FILE_PATH = "custom_4way.net.xml"
ROUTE_FILE_PATH = "custom_4way.rou.xml"


class CustomObservationFunction(ObservationFunction):
    """
    **Need to modify:** I Copy/pasted the default ObservationFunction for traffic
    signals just to test that it would work.
    """

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [
            1 if self.ts.green_phase == i else 0
            for i in range(self.ts.num_green_phases)
        ]  # one-hot encoding
        min_green = [
            (
                0
                if self.ts.time_since_last_phase_change
                < self.ts.min_green + self.ts.yellow_time
                else 1
            )
        ]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(
                self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32
            ),
            high=np.ones(
                self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32
            ),
        )


def custom_reward_fn(ts: TrafficSignal):
    """
    **Need to modify:** Copy/pasted from TrafficSignal._diff_waiting_time_reward(self)
    just to test that this would work.
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = ts.last_ts_waiting_time - ts_wait
    ts.last_ts_waiting_time = ts_wait
    return reward


"""
Register the custom rewards function using the dedicated TrafficSignal static method.
Register the customized environment to the Gymnasium API. First need to set valid path 
for net_file and route file (create them if they don't exist).
"""
TrafficSignal.register_reward_fn(custom_reward_fn)
register(
    id=CUSTOM_ENV_ID,
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={
        "single_agent": True,
        "net_file": NET_FILE_PATH,  # NEED TO CREATE
        "route_file": ROUTE_FILE_PATH,  # NEED TO CREATE
        "reward_fn": custom_reward_fn,  # NEED TO IMPLEMENT ABOVE
        "observation_class": CustomObservationFunction,  # NEED TO IMPLEMENT ABOVE
    },
)
