"""-------------------------------------------------------------------------------------
File: generate_route_file.py
Description: Script to generate normalized route file variations that are randomly 
    asymmetric for use in the training and evaluation of RL agorithms. With symmetric 
    files, the best algorithm is random, whereas with assymetric traffic flows, then 
    the algorihms will hopefuly learn useful patterns.
Author: Julien Lariviere-Chartier
-------------------------------------------------------------------------------------"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

# Directory for the network files
NETWORK_DIR = Path("demo-intersection")
BASE_ROUTE_FILE = NETWORK_DIR / "demo-intersection.rou.xml"

# Where to output the assymetric files
ASSYMETRIC_ROUTES_DIR = NETWORK_DIR / "asymmetric_route_files"

# All the traffic flow directions
FLOW_DIRECTIONS = [
    "N_S",
    "N_E",
    "N_W",
    "S_N",
    "S_E",
    "S_W",
    "E_N",
    "E_S",
    "E_W",
    "W_N",
    "W_S",
    "W_E",
]

# Number of episodes
NUM_EPISODES = 50

# Config to keep the load fair between episodes
FLOW_CONFIG = {
    "car": {"min": 0.1, "max": 0.4, "sum": 2.4},
    "ped": {"min": 0.02, "max": 0.1, "sum": 0.6},
}

def get_route_file_name(route_files_dir:Path, variation_index:int) -> Path:
    # Helper to get the route file name
    return route_files_dir / f"routes_{variation_index:02d}.xml"

def generate_route_files(
    base_file: Path = BASE_ROUTE_FILE,
    num_variations: int = NUM_EPISODES,
    flow_directions: list[str] = FLOW_DIRECTIONS,
    flow_config: dict = FLOW_CONFIG,
    output_dir: Path = ASSYMETRIC_ROUTES_DIR,
):
    """Script to generate 
    """
    # Create the output dir if not created already
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Open the base rou.xml file
    tree = ET.parse(base_file)
    root_template = tree.getroot()

    # For each variation (each episode)
    for variation in range(num_variations):

        # Copy the template for modifying it
        root = ET.fromstring(ET.tostring(root_template))

        """Generate normalized probabilities per config to keep the different files 
        fair. Generate 1 probability per flow type and per direction. They all must sum
        to the same value (per flow type) to keep things fair across episodes."""
        probs = {}
        for flow_type, config in flow_config.items():
            probs[flow_type] = np.random.uniform(
                config["min"], config["max"], len(flow_directions)
            )
            probs[flow_type] *= config["sum"] / probs[flow_type].sum()

        # Update flow in each direction for both types of flows (flow, personFLow)
        for index, direction in enumerate(flow_directions):

            # Find the xml element and set the flow value for each type
            car_flow = root.find(f".//flow[@id='car_{direction}']")
            car_flow.set("probability", f"{probs['car'][index]:.2f}")

            ped_flow = root.find(f".//personFlow[@id='ped_{direction}']")
            ped_flow.set("probability", f"{probs['ped'][index]:.2f}")

        # Write the route variation file
        ET.ElementTree(root).write(
            get_route_file_name(output_dir, variation), encoding="utf-8", xml_declaration=True
        )


if __name__ == "__main__":
    generate_route_files()
