"""-------------------------------------------------------------------------------------
File: test.py
Description: First test to see if the SUMO-RL custom environment and code pulled from
GitHub can be used.
-------------------------------------------------------------------------------------"""

import gymnasium as gym
import sumo_rl
from time import sleep
from custom_env import CUSTOM_ENV_ID  # This import registers the custom environment

# Test environment name (as registererd in Gymnasium)
TEST_ENV_ID = "sumo-rl-v0"

# Location of files for single intersection (adapt path to your sumo-rl installation)
SINGLE_INTERSECTION_DIR = "/opt/sumo-rl/sumo_rl/nets/single-intersection/"

# GUI parameters
USE_GUI = True
STEP_SLEEP_DELAY = 1.0

def test_sumo_rl(env: gym.Env):
    """
    Function to test the SUMO-RL library with single intersection network/route
    files.
    """

    # Reset the environment, extract initial observation and information
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Init the step index and terminated/truncated flags
    step_index = 0
    terminated = False
    truncated = False

    # Loop until episode is done
    while not (terminated or truncated):

        # Choose random action, take it and observe nenxt state, rewards, etc.
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Print step info
        print(f"\n# Step {step_index}:")
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        # Increment step
        step_index += 1

        # Sleep so GUI is watchable
        if USE_GUI:
            sleep(STEP_SLEEP_DELAY)

    env.close()


if __name__ == "__main__":

    # Create the default test Gym Env
    """ env = gym.make(
        TEST_ENV_ID,
        num_seconds=200,
        virtual_display=(1920, 1800),
        use_gui=USE_GUI,
        net_file=SINGLE_INTERSECTION_DIR + "single-intersection.net.xml",
        route_file=SINGLE_INTERSECTION_DIR + "single-intersection.rou.xml",
    ) """

    # Create the custom Gym Env
    env = gym.make(
        CUSTOM_ENV_ID,
        num_seconds=3600,
        virtual_display=(1920, 1800),
        use_gui=True,
        #net_file=SINGLE_INTERSECTION_DIR + "single-intersection.net.xml", # Comment this to test with the custom map
        #route_file=SINGLE_INTERSECTION_DIR + "single-intersection.rou.xml", # Comment this to test with the custom map
    )
    
    # Run the test on teh selected environment
    test_sumo_rl(env)
