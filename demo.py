"""-------------------------------------------------------------------------------------
File: demo.py
Description: Enhanced demo script for Oct 27 presentation. This script provides
clearer output and statistics compared to test.py, making it better suited for
demonstration purposes.
-------------------------------------------------------------------------------------"""

import gymnasium as gym
import sumo_rl
import numpy as np
from time import sleep
from custom_env import CUSTOM_ENV_ID

# Demo configuration
USE_GUI = True
DEMO_DURATION_SECONDS = 300  # 300 simulation seconds
STEP_SLEEP_DELAY = 0  # 0 seconds for smoother animation
PRINT_EVERY_N_STEPS = 5  # Print every 10 steps for progress updates

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_section(text):
    """Print a formatted section."""
    print(f"\n--- {text} ---")

def print_observation_details(obs, env):
    """Print detailed breakdown of observation."""
    print_section("Observation Breakdown")
    
    # Get the traffic signal (assuming single agent)
    ts = list(env.unwrapped.traffic_signals.values())[0]
    
    idx = 0
    
    # Phase ID
    num_phases = ts.num_green_phases
    phase_id = obs[idx:idx+num_phases]
    current_phase = np.argmax(phase_id)
    print(f"  Current Phase: {current_phase} (out of {num_phases} phases)")
    idx += num_phases
    
    # Min green flag
    min_green_ok = obs[idx]
    print(f"  Min Green Time Elapsed: {'Yes' if min_green_ok else 'No'}")
    idx += 1
    
    # Current time (normalized)
    current_time = obs[idx]
    time_seconds = current_time * 3600.0  # Convert back from normalized
    print(f"  Simulation Time: {time_seconds:.1f}s ({current_time*100:.1f}% complete)")
    idx += 1
    
    # Vehicle lanes
    num_vehicle_lanes = len(ts.lanes)
    vehicle_density = obs[idx:idx+num_vehicle_lanes]
    idx += num_vehicle_lanes
    vehicle_queue = obs[idx:idx+num_vehicle_lanes]
    idx += num_vehicle_lanes
    
    print(f"  Vehicle Lanes ({num_vehicle_lanes} lanes):")
    print(f"    Avg Density: {np.mean(vehicle_density):.3f}")
    print(f"    Avg Queue:   {np.mean(vehicle_queue):.3f}")
    
    # Pedestrian lanes
    num_ped_lanes = len(ts.ped_lanes)
    ped_density = obs[idx:idx+num_ped_lanes]
    idx += num_ped_lanes
    ped_queue = obs[idx:idx+num_ped_lanes]
    idx += num_ped_lanes
    
    print(f"  Pedestrian Lanes ({num_ped_lanes} lanes):")
    print(f"    Avg Density: {np.mean(ped_density):.3f}")
    print(f"    Avg Queue:   {np.mean(ped_queue):.3f}")

def run_demo(env: gym.Env):
    """
    Run the demo with enhanced statistics and clearer output.
    """
    
    print_header("TRAFFIC SIGNAL CONTROL DEMO - Team 24")
    print("\nProject: Reinforcement Learning for Traffic Signal Control")
    print("Environment: Custom 4-way intersection with pedestrians")
    print(f"Duration: {DEMO_DURATION_SECONDS} simulation seconds")
    print("\nAgent: Random (demonstrating environment interaction)")
    
    # Display environment information
    print_section("Environment Information")
    obs_shape = env.observation_space.shape[0]
    obs_low = env.observation_space.low[0]
    obs_high = env.observation_space.high[0]
    print(f"  Observation Space: {obs_shape}-dimensional continuous (range: [{obs_low:.1f}, {obs_high:.1f}])")
    print(f"  Action Space: Discrete with {env.action_space.n} actions (traffic signal phases)")
    print(f"  → Agent can choose from {env.action_space.n} different traffic light configurations")
    
    # Reset environment
    print_section("Resetting Environment")
    obs, info = env.reset()
    print("✓ Environment reset successfully")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial info: {info}")
    
    # Print initial observation details
    print_observation_details(obs, env)
    
    # Statistics tracking
    stats = {
        'total_reward': 0.0,
        'steps': 0,
        'actions_taken': [],
        'rewards': [],
        'min_reward': float('inf'),
        'max_reward': float('-inf'),
    }
    
    # Main loop
    print_header("STARTING SIMULATION")
    print("Watch the SUMO GUI to see vehicles and pedestrians moving!")
    print("Traffic lights will change based on the agent's actions.\n")
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Choose random action
        action = env.action_space.sample()
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update statistics
        stats['steps'] += 1
        stats['total_reward'] += reward
        stats['actions_taken'].append(action)
        stats['rewards'].append(reward)
        stats['min_reward'] = min(stats['min_reward'], reward)
        stats['max_reward'] = max(stats['max_reward'], reward)
        
        # Print progress (less frequently)
        if stats['steps'] % PRINT_EVERY_N_STEPS == 0:
            print(f"\nStep {stats['steps']:4d} | Action: {action} | Reward: {reward:8.3f} | "
                  f"Cumulative Reward: {stats['total_reward']:8.3f}")
            # Print observation details every N steps
            print_observation_details(obs, env)
        
        # Sleep for visualization
        if USE_GUI:
            sleep(STEP_SLEEP_DELAY)
    
    # Print final statistics
    print_header("SIMULATION COMPLETE - STATISTICS")
    
    print_section("General Statistics")
    print(f"  Total Steps: {stats['steps']}")
    print(f"  Episode Ended: {'Terminated' if terminated else 'Truncated'}")
    
    print_section("Reward Statistics")
    print(f"  Total Reward: {stats['total_reward']:.3f}")
    print(f"  Average Reward per Step: {np.mean(stats['rewards']):.3f}")
    print(f"  Minimum Reward: {stats['min_reward']:.3f}")
    print(f"  Maximum Reward: {stats['max_reward']:.3f}")
    print(f"  Reward Std Dev: {np.std(stats['rewards']):.3f}")
    
    print_section("Action Distribution")
    unique_actions, counts = np.unique(stats['actions_taken'], return_counts=True)
    for action, count in zip(unique_actions, counts):
        percentage = (count / stats['steps']) * 100
        print(f"  Phase {action}: {count:4d} times ({percentage:5.1f}%)")
    
    # Traffic signal specific statistics
    ts = list(env.unwrapped.traffic_signals.values())[0]
    print_section("Traffic Signal Statistics")
    print(f"  Total Vehicles Queued (final): {ts.get_total_queued()}")
    print(f"  Total Pedestrians Queued (final): {ts.get_total_pedestrian_queued()}")
    
    print_header("DEMO COMPLETE")
    print("Thank you for watching our Traffic Signal Control demonstration!")
    print("Next steps: Implement Q-learning, PPO, Max-Pressure, and AGGM algorithms\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("  COMP 4010 - Team 24 - Traffic Signal Control Environment Demo")
    print("=" * 80 + "\n")
    
    # Create the custom environment
    print("Creating custom environment...")
    env = gym.make(
        CUSTOM_ENV_ID,
        num_seconds=DEMO_DURATION_SECONDS,
        delta_time=5,  # Action every 5 sim seconds = smoother, more frequent updates
        virtual_display=(1920, 1080),
        use_gui=USE_GUI,
        additional_sumo_cmd="--delay 100",  # SUMO GUI delay: 100ms = smooth real-time animation
    )
    print("✓ Environment created successfully\n")
    
    # Run the demo
    try:
        run_demo(env)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        env.close()
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        env.close()

