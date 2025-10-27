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
from sumo_rl.environment.env import SumoEnvironment, TrafficSignal, ObservationFunction, LIBSUMO
import sumolib
import traci
import time

# Id of the custom environment registered to Gymnasium API
CUSTOM_ENV_ID = "custom-tsc-env-v0"

"""
The traffic scenario map. Need to create those 2 files using the `net` GUI tool in 
${SUMO_HOME}/tools)"""
DEMO_DIR = "demo-intersection/"
NET_FILE_PATH = DEMO_DIR + "demo-intersection.net.xml"
ROUTE_FILE_PATH = DEMO_DIR + "demo-intersection.rou.xml"

# Has less objects, perfect for GUI demo
LITE_ROUTE_FILE_PATH = DEMO_DIR + "demo-intersection-lite.rou.xml"

"""Number of seconds to wait before starting the GUI simulation (needed for 
object initialization)"""
START_SIMULATION_DELAY = 2

class CustomSumoEnvironment(SumoEnvironment):
    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            
            # Add a delay to start the simulation so the objects are loaded in memory
            time.sleep(START_SIMULATION_DELAY)
            try:
                if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                    traci.gui.DEFAULT_VIEW = "View #0"
                self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            except Exception as e:
                print(f"Warning: could not set GUI schema: {e}")



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
""" TrafficSignal.register_reward_fn(custom_reward_fn)
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
) """

TrafficSignal.register_reward_fn(custom_reward_fn)
register(
    id=CUSTOM_ENV_ID,
    entry_point="custom_env:CustomSumoEnvironment",
    kwargs={
        "single_agent": True,
        "net_file": NET_FILE_PATH,
        "route_file": ROUTE_FILE_PATH,
        "reward_fn": custom_reward_fn,  # NEED TO IMPLEMENT ABOVE
        "observation_class": CustomObservationFunction,  # NEED TO IMPLEMENT ABOVE
    },
)