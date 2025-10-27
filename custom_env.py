"""-------------------------------------------------------------------------------------
File: custom_env.py
Description: Definition of a custom Gymnasium env, with ObservationFunction class and a 
custom reward function for the Env Demo project deliverable. Importing this file 
registers the custom environment to the Gymnasium API. The NET_FILE_PATH and 
ROUTE_FILE_PATH have to be designed for this to work.
-------------------------------------------------------------------------------------"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from sumo_rl.environment.env import (
    SumoEnvironment,
    TrafficSignal,
    ObservationFunction,
    LIBSUMO,
)
import sumolib
import traci
import time
from typing import Union, Optional
from typing_extensions import Callable
# Id of the custom environment registered to Gymnasium API
CUSTOM_ENV_ID = "custom-tsc-env-v0"

# The traffic scenario map. Need to create those 2 files using the `netedit` tool
DEMO_DIR = "demo-intersection/"
NET_FILE_PATH = DEMO_DIR + "demo-intersection.net.xml"
ROUTE_FILE_PATH = DEMO_DIR + "demo-intersection.rou.xml"

"""Number of seconds to wait before starting the GUI simulation (needed for 
object initialization)"""
START_SIMULATION_DELAY = 2


class CustomTrafficSignal(TrafficSignal):
    
    MIN_PED_GAP = 0.5
    
    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, list],
        reward_weights: list[float],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo = sumo

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()
        
        # Separate vehicles and pedestrians
        all_lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))
        self.lanes = [lane for lane in all_lanes if not lane.startswith(":")]
        self.ped_lanes = [lane for lane in all_lanes if lane.startswith(":")]
        
        # Only compute out_lanes for vehicles
        out_lanes = set(link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link)
        self.out_lanes = [lane for lane in out_lanes if not lane.startswith(":")]

        # Recompute lengths including ped lanes if needed
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes + self.ped_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def get_pedestrian_density(self) -> list[float]:
        """Returns the density [0,1] of pedestrians in incoming pedestrian lanes."""
        densities = []
        for lane in self.ped_lanes:
            # Count pedestrians on this lane
            ped_on_lane = sum(
                1 for ped_id in self.sumo.person.getIDList() if self.sumo.person.getLaneID(ped_id) == lane
            )
            # Compute density relative to lane length
            density = ped_on_lane / max(1, self.lanes_length[lane] / self.MIN_PED_GAP)
            densities.append(min(1, density))
        return densities

    def get_pedestrian_queue(self) -> list[float]:
        """Returns the queue [0,1] of pedestrians in incoming pedestrian lanes."""
        queues = []
        for lane in self.ped_lanes:
            ped_on_lane = sum(
                1 for ped_id in self.sumo.person.getIDList()
                if self.sumo.person.getLaneID(ped_id) == lane and self.sumo.person.getSpeed(ped_id) < 0.1
            )
            queue = ped_on_lane / max(1, self.lanes_length[lane] / self.MIN_PED_GAP)
            queues.append(min(1, queue))
        return queues

    def get_total_pedestrian_queued(self) -> int:
        """Returns the total number of pedestrians waiting to cross."""
        return sum(self.sumo.lane.getLastStepHaltingPersonNumber(lane) for lane in self.ped_lanes)

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles and pedestrians halting in the intersection."""
        total_vehicles = super().get_total_queued()  # sums over self.lanes (vehicles)
        total_peds = sum(
            1 for lane in self.ped_lanes
            for ped_id in self.sumo.person.getIDList()
            if self.sumo.person.getLaneID(ped_id) == lane and self.sumo.person.getSpeed(ped_id) < 0.1
        )
        return total_vehicles + total_peds


class CustomSumoEnvironment(SumoEnvironment):
    """
    Custom version of the SumoEnvironment which overrides the _start_simulation() method
    to allow for arbitrary number of pedestrians and vehicles to spawn in the simulation
    by adding a sleep(delay) before the simulation variables are accessed by the code.
    This fix gives reasonable time for the simulation to initialize and prevents
    crashes.
    """

    def _build_traffic_signals(self, conn):
        """Build CustomTrafficSignal objects that also keep track of pedestrians"""
        if not isinstance(self.reward_fn, dict):
            self.reward_fn = {ts: self.reward_fn for ts in self.ts_ids}

        self.traffic_signals = {
            ts: CustomTrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                self.enforce_max_green,
                self.begin_time,
                self.reward_fn[ts],
                self.reward_weights,
                conn,
            )
            for ts in self.ts_ids
        }

    def _start_simulation(self):
        """This method starts the simulation GUI but properly waits before setting 
        traci.gui.DEFAULT_VIEW for the simulation to have fully initialized 
        (added a sleep(delay)). This prevents a crash when too many sim objects."""
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
                sumo_cmd.extend(
                    [
                        "--window-size",
                        f"{self.virtual_display[0]},{self.virtual_display[1]}",
                    ]
                )
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
                if "DEFAULT_VIEW" not in dir(
                    traci.gui
                ):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                    traci.gui.DEFAULT_VIEW = "View #0"
                self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            except Exception as e:
                print(f"Warning: could not set GUI schema: {e}")


class CustomObservationFunction(ObservationFunction):
    """
    **Need to modify:** I Copy/pasted the default ObservationFunction for traffic
    signals just to test that it would work.
    """

    def __init__(self, ts: CustomTrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        vehicle_density = self.ts.get_lanes_density()
        vehicle_queue = self.ts.get_lanes_queue()

        ped_density = self.ts.get_pedestrian_density()
        ped_queue = self.ts.get_pedestrian_queue()

        observation = np.array(phase_id + min_green + vehicle_density + vehicle_queue + ped_density + ped_queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        total_vehicle_lanes = len(self.ts.lanes)
        total_ped_lanes = len(self.ts.ped_lanes)

        obs_len = self.ts.num_green_phases + 1 + 2 * (total_vehicle_lanes + total_ped_lanes)

        return spaces.Box(low=np.zeros(obs_len, dtype=np.float32),
                        high=np.ones(obs_len, dtype=np.float32))



def custom_reward_fn(ts: CustomTrafficSignal):
    """
    **Need to modify:** Copy/pasted from TrafficSignal._diff_waiting_time_reward(self)
    just to test that this would work.
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = ts.last_ts_waiting_time - ts_wait
    ts.last_ts_waiting_time = ts_wait
    return reward


# Register the reward function to the CustomTrafficSignal class
#TrafficSignal.register_reward_fn(custom_reward_fn)
CustomTrafficSignal.register_reward_fn(custom_reward_fn)
"""Register the custom environment, with custom observation, reward function, 
and scenario files) to the Gymnasium API"""
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
