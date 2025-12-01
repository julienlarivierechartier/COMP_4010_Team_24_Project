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
import xml.etree.ElementTree as ET
from pathlib import Path
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

# Weight for pedestrian waiting time in pressure calculation
DEAFULT_PED_WAIT_WEIGHT = 0.1

# Default pedestrian crossing distance in meters
DEFAULT_CROSSING_DISTANCE = 16.0  # meters

# Yellow_time (3s) is the main buffer for pedestrians crossing
# We don't add this to min_green calculation to avoid double-buffering
PEDESTRIAN_SAFETY_MARGIN = 0.5  # seconds - only used in runtime safety checks, not min_green calc


def get_pedestrian_speed_from_route_file(route_file_path: str) -> float:
    """
    Reads the pedestrian desiredMaxSpeed from a route file.
    Returns the speed in m/s, or a default value if not found.
    """
    try:
        tree = ET.parse(route_file_path)
        root = tree.getroot()
        
        # Look for vType with id="pedestrian" or vClass="pedestrian"
        for vtype in root.findall('.//vType'):
            vclass = vtype.get('vClass', '')
            if vclass == 'pedestrian':
                speed_str = vtype.get('desiredMaxSpeed')
                if speed_str:
                    return float(speed_str)
        
        # If not found, return default (standard pedestrian speed)
        return 2.0  # m/s - standard pedestrian speed in route files
    except Exception as e:
        print(f"Warning: Could not read pedestrian speed from {route_file_path}: {e}")
        return 2.0  # Default to standard pedestrian speed


def calculate_min_green_time(
    crossing_distance: float,
    pedestrian_speed: float,
    safety_margin: float = PEDESTRIAN_SAFETY_MARGIN,
    yellow_time: int = 3
) -> int:
    """
    Calculates the minimum green time needed for pedestrians to cross safely.
    min_green is set to crossing_time + minimal safety_margin.
    Yellow_time (3s) serves as the main buffer/slack for pedestrians still crossing.
    
    Args:
        crossing_distance: Distance to cross in meters
        pedestrian_speed: Pedestrian walking speed in m/s
        safety_margin: Minimal safety margin in seconds (yellow_time is the main buffer)
        yellow_time: Yellow phase duration in seconds (used for info, not calculation)
    
    Returns:
        Minimum green time in seconds (rounded up to nearest integer)
    """
    if pedestrian_speed <= 0:
        # Default to 2.0 m/s which is the standard pedestrian speed in route files
        pedestrian_speed = 2.0
    
    crossing_time = crossing_distance / pedestrian_speed
    
    # min_green = crossing_time (yellow_time provides the buffer)
    # Yellow_time (3s) is the buffer - provides slack for pedestrians still crossing
    # We don't add safety_margin here to avoid double-buffering
    min_green = int(np.ceil(crossing_time))
    
    # Ensure minimum is at least 5 seconds for basic traffic flow
    return max(min_green, 5)


def get_crossing_distance_from_network(net_file_path: str) -> float:
    """
    Reads the crossing distance from the network file.
    Returns the length of the first crossing edge found, or default.
    """
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
        
        # Look for edges with function="crossing"
        for edge in root.findall('.//edge'):
            if edge.get('function') == 'crossing':
                for lane in edge.findall('.//lane'):
                    length_str = lane.get('length')
                    if length_str:
                        return float(length_str)
        
        return DEFAULT_CROSSING_DISTANCE
    except Exception as e:
        print(f"Warning: Could not read crossing distance from {net_file_path}: {e}")
        return DEFAULT_CROSSING_DISTANCE

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
        ped_wait_weight: float = DEAFULT_PED_WAIT_WEIGHT,
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
        
        # Init weight for pedestrian waiting time in pressure calculation
        self.ped_wait_weight = ped_wait_weight

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

    # Compute the list of absolute phase indices used for green actions
    def _get_green_phase_indices(self) -> list[int]:
        logic = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        g = getattr(self, "green_phases", None)
        if g:
            g_list = list(g)
            if isinstance(g_list[0], int):
                return g_list
            # Map Phase objects to their indices by state
            state_to_idx = {ph.state: i for i, ph in enumerate(logic.phases)}
            idxs = [state_to_idx.get(getattr(ph, "state", None)) for ph in g_list]
            idxs = [i for i in idxs if i is not None]
            if idxs:
                return idxs
        # Fallback: detect phases with greens and no yellow
        green_idxs = []
        for idx, ph in enumerate(logic.phases):
            s = ph.state
            if any(c in "gG" for c in s) and not any(c in "yY" for c in s):
                green_idxs.append(idx)
        return green_idxs

    # Compute instantaneous pressure for a given absolute phase index
    def _phase_pressure(self, abs_phase_idx: int) -> float:
        logic = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        state = logic.phases[abs_phase_idx].state
        links = self.sumo.trafficlight.getControlledLinks(self.id)

        pressure = 0.0
        for i, link_list in enumerate(links):
            if not link_list:
                continue
            ch = state[i]
            if ch not in ("g", "G"):
                continue
            in_lane = link_list[0][0]
            out_lane = link_list[0][1]

            # Vehicles: upstream queue minus downstream queue
            if not in_lane.startswith(":"):
                up_q = float(self.sumo.lane.getLastStepHaltingNumber(in_lane))
                down_q = 0.0
                if out_lane and not out_lane.startswith(":"):
                    down_q = float(self.sumo.lane.getLastStepHaltingNumber(out_lane))
                pressure += (up_q - down_q)
            else:
                # Pedestrians: count waiting + small bonus for wait time
                ped_ids = [pid for pid in self.sumo.person.getIDList()
                           if self.sumo.person.getLaneID(pid) == in_lane and self.sumo.person.getSpeed(pid) < 0.1]
                ped_count = float(len(ped_ids))
                avg_wait = 0.0
                if ped_ids:
                    waits = [self.sumo.person.getWaitingTime(pid) for pid in ped_ids]
                    avg_wait = float(sum(waits)) / len(waits)
                # Use the configurable pedestrain wait weight in the pressure calculation
                pressure += ped_count + self.ped_wait_weight * avg_wait

        return pressure

    # Select the green action index that maximizes pressure
    def select_max_pressure_action(self) -> int:
        if self.is_yellow:
            return self.green_phase
        if self.time_since_last_phase_change < self.min_green:
            return self.green_phase

        # Check if pedestrians are currently crossing and need more time
        # Prevent phase change if pedestrians won't have enough time to finish crossing
        # Pedestrians can use both green AND yellow time to cross
        if hasattr(self.env, 'pedestrian_crossing_time'):
            # Calculate remaining green time (including potential extension beyond min_green)
            # We check if we're still in the min_green period or if we've extended
            time_in_green = self.time_since_last_phase_change
            
            # Calculate how much time pedestrians need to finish crossing
            # Yellow_time is the buffer, so we just need crossing_time
            required_time = self.env.pedestrian_crossing_time
            
            # Check if any pedestrians are currently on crossing lanes (actively crossing)
            pedestrians_crossing = False
            for lane in self.ped_lanes:
                ped_ids = [pid for pid in self.sumo.person.getIDList() 
                          if self.sumo.person.getLaneID(pid) == lane 
                          and self.sumo.person.getSpeed(pid) > 0.1]
                if ped_ids:
                    pedestrians_crossing = True
                    break
            
            # If pedestrians are crossing, ensure we have enough time remaining
            # Total available time = remaining_green + yellow_time
            # We need: remaining_green + yellow_time >= crossing_time + safety_margin
            if pedestrians_crossing:
                # Calculate remaining time before we could change phase
                # If we're past min_green, we could change at next delta_time
                if time_in_green >= self.min_green:
                    # We could change phase soon (at next delta_time)
                    # Remaining green = delta_time (time until next action)
                    # Total available = delta_time + yellow_time
                    remaining_green = self.delta_time
                    total_available_time = remaining_green + self.yellow_time
                    if total_available_time < required_time:
                        # Not enough time (green + yellow), keep current phase
                        return self.green_phase
                else:
                    # Still in min_green period
                    # Remaining green = min_green - time_in_green
                    # Total available = remaining_green + yellow_time
                    remaining_green = self.min_green - time_in_green
                    total_available_time = remaining_green + self.yellow_time
                    if total_available_time < required_time:
                        # Not enough time (green + yellow), keep current phase
                        return self.green_phase

        green_idxs = self._get_green_phase_indices()
        best_a = self.green_phase
        best_p = -1e18
        for a, abs_idx in enumerate(green_idxs):
            p = self._phase_pressure(abs_idx)
            if p > best_p:
                best_p = p
                best_a = a
        return best_a

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
        return sum(
            1 for lane in self.ped_lanes
            for ped_id in self.sumo.person.getIDList()
            if self.sumo.person.getLaneID(ped_id) == lane and self.sumo.person.getSpeed(ped_id) < 0.1
        )

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

        # Calculate pedestrian crossing parameters if not already done
        if not hasattr(self, 'pedestrian_crossing_time'):
            crossing_distance = get_crossing_distance_from_network(self._net)
            pedestrian_speed = get_pedestrian_speed_from_route_file(self._route)
            calculated_min_green = calculate_min_green_time(
                crossing_distance,
                pedestrian_speed,
                PEDESTRIAN_SAFETY_MARGIN,
                self.yellow_time
            )
            if self.min_green < calculated_min_green:
                self.min_green = calculated_min_green
            self.pedestrian_crossing_time = crossing_distance / pedestrian_speed
            self.pedestrian_speed = pedestrian_speed

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
        
        # Calculate pedestrian crossing parameters
        crossing_distance = get_crossing_distance_from_network(self._net)
        pedestrian_speed = get_pedestrian_speed_from_route_file(self._route)
        calculated_min_green = calculate_min_green_time(
            crossing_distance, 
            pedestrian_speed, 
            PEDESTRIAN_SAFETY_MARGIN,
            self.yellow_time
        )
        
        # Update min_green if it was not set or is too small
        if self.min_green < calculated_min_green:
            print(f"Updating min_green from {self.min_green} to {calculated_min_green} "
                  f"(based on crossing distance {crossing_distance}m, "
                  f"pedestrian speed {pedestrian_speed}m/s)")
            self.min_green = calculated_min_green
        
        # Calculate minimum crossing time for pedestrians
        min_crossing_time = crossing_distance / pedestrian_speed
        
        # Store crossing time for use in pedestrian control
        self.pedestrian_crossing_time = min_crossing_time
        self.pedestrian_speed = pedestrian_speed
        
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
            # Added this to prevent pedestrian jams
            "--pedestrian.striping.jamtime", "600",
            "--pedestrian.striping.jamtime.crossing", "60",
            # Ensure pedestrians respect traffic light timing
            # The min_green time ensures pedestrians have enough time to cross
            # Additional logic in select_max_pressure_action prevents phase changes
            # when pedestrians are still crossing
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
    Custom observation function that includes both vehicles and pedestrians.
    Includes: phase ID, min green flag, current time, vehicle queue, and pedestrian queue.
    """

    def __init__(self, ts: CustomTrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the observation including pedestrians."""
        # Current traffic signal phase (one-hot encoded)
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        
        # Whether minimum green time has elapsed (yellow_time is separate, not part of min_green)
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green else 1]
        
        # Current simulation time (normalized to [0,1] for 1 hour episode)
        current_time = [self.ts.env.sim_step / 3600.0]

        # Vehicle lanes: density and queue
        vehicle_density = self.ts.get_lanes_density()
        vehicle_queue = self.ts.get_lanes_queue()

        # Pedestrian lanes: density and queue
        ped_density = self.ts.get_pedestrian_density()
        ped_queue = self.ts.get_pedestrian_queue()

        observation = np.array(
            phase_id + min_green + current_time + 
            vehicle_density + vehicle_queue + 
            ped_density + ped_queue, 
            dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        total_vehicle_lanes = len(self.ts.lanes)
        total_ped_lanes = len(self.ts.ped_lanes)

        # phase_id + min_green + current_time + 2*(vehicles) + 2*(pedestrians)
        obs_len = self.ts.num_green_phases + 1 + 1 + 2 * (total_vehicle_lanes + total_ped_lanes)

        return spaces.Box(low=np.zeros(obs_len, dtype=np.float32),
                        high=np.ones(obs_len, dtype=np.float32) * 1000)  # Allow values > 1 for time



def custom_reward_fn(ts: CustomTrafficSignal):
    """
    Custom reward function that penalizes waiting time and queue length
    for both vehicles and pedestrians with equal weighting.
    
    Returns negative reward for:
    - Vehicle waiting time (differential)
    - Total queue length (vehicles + pedestrians weighted equally)
    """
    # Component 1: Vehicle waiting time (differential - like original)
    vehicle_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    wait_penalty = ts.last_ts_waiting_time - vehicle_wait
    ts.last_ts_waiting_time = vehicle_wait
    
    # Component 2: Queue length penalty (vehicles + pedestrians weighted equally)
    total_queued = ts.get_total_queued()  # Includes both vehicles and pedestrians
    queue_penalty = -0.01 * total_queued
    
    # Combined reward (equal weighting for vehicles and pedestrians)
    reward = wait_penalty + queue_penalty
    return reward


# Register the reward function to the CustomTrafficSignal class
#TrafficSignal.register_reward_fn(custom_reward_fn)
CustomTrafficSignal.register_reward_fn(custom_reward_fn)
"""Register the custom environment, with custom observation, reward function, 
and scenario files) to the Gymnasium API"""

# Calculate default min_green based on pedestrian crossing requirements
_default_crossing_distance = get_crossing_distance_from_network(NET_FILE_PATH)
_default_pedestrian_speed = get_pedestrian_speed_from_route_file(ROUTE_FILE_PATH)
_default_yellow_time = 3  # Default yellow time (will be overridden if specified in register)
_default_min_green = calculate_min_green_time(
    _default_crossing_distance,
    _default_pedestrian_speed,
    PEDESTRIAN_SAFETY_MARGIN,
    _default_yellow_time
)

# Ensure delta_time is compatible with min_green
# delta_time should be <= min_green and ideally a factor of min_green
_default_delta_time = 5
if _default_min_green > _default_delta_time:
    # If min_green is larger, ensure delta_time divides evenly or is at least half
    if _default_min_green % _default_delta_time != 0:
        # Adjust delta_time to be compatible (use a value that works well)
        # Keep delta_time at 5 if min_green is reasonable, otherwise adjust
        if _default_min_green <= 15:
            _default_delta_time = 5
        elif _default_min_green <= 20:
            _default_delta_time = 5  # Still works, just means actions happen every 5s
        else:
            _default_delta_time = max(5, _default_min_green // 3)  # Roughly 1/3 of min_green

print(f"Pedestrian crossing parameters: distance={_default_crossing_distance}m, "
      f"speed={_default_pedestrian_speed}m/s, min_green={_default_min_green}s, "
      f"delta_time={_default_delta_time}s")

register(
    id=CUSTOM_ENV_ID,
    entry_point="custom_env:CustomSumoEnvironment",
    kwargs={
        "single_agent": True,
        "net_file": NET_FILE_PATH,
        "route_file": ROUTE_FILE_PATH,
        "reward_fn": custom_reward_fn,  
        "observation_class": CustomObservationFunction,  
        "num_seconds":3600,
        
        # Calculate these based on pedestrian crossing requirements
        # min_green is calculated to ensure pedestrians have enough time to cross
        "delta_time": _default_delta_time,
        "yellow_time": 3,
        "min_green": _default_min_green,  # Will be updated dynamically if route file has different speed
        "max_green": 30,
    },
)
