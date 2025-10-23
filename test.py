"""-------------------------------------------------------------------------------------
File: test.py
Description: First test to see if the SUMO-RL custom environment and code pulled from
GitHub can be used.
-------------------------------------------------------------------------------------"""

import gymnasium as gym
import sumo_rl

# Location of files for single intersection (adapt path to your sumo-rl installation)
SINGLE_INTERSECTION_DIR = "/opt/sumo-rl/sumo_rl/nets/single-intersection/"


def test_sumo_rl():
    """
    Function to test the SUMO-RL library with single intersection network/route
    files.
    """

    # Create the custom Gym Env
    env = gym.make(
        "sumo-rl-v0",
        num_seconds=200,
        use_gui=False,
        net_file=SINGLE_INTERSECTION_DIR + "single-intersection.net.xml",
        route_file=SINGLE_INTERSECTION_DIR + "single-intersection.rou.xml",
    )

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

    env.close()


if __name__ == "__main__":
    test_sumo_rl()
