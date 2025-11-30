"""For enhanced performance (using libsumo C++ backend), then call 
"export LIBSUMO_AS_TRACI=1" before launching this script.
 """

from itertools import product
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import time
from typing import Optional

# Import the algorithms (agents)
from algorithms import (
    BaseAlgorithm, 
    PPOAgent, 
    QLearningAgent, 
    MaxPressureAgent,
    RandomAgent,
    FixedTimeAgent,
    DQNAgent
)

import gymnasium as gym
from custom_env import CUSTOM_ENV_ID
from generate_route_files import (
    NUM_EPISODES, 
    ASSYMETRIC_ROUTES_DIR, 
    BASE_ROUTE_FILE,
    get_route_file_name,
)

# Set the seed for testing or set to None for results generation.
SUMO_SEED = None

# Eval route files
RANDOM_SEED = 32

# Reference for the algorithms evaluated
ALGORITHMS = {
    "ppo": PPOAgent,
    "max_pressure": MaxPressureAgent,
    "random": RandomAgent,
    "fixed_time": FixedTimeAgent,
    "dqn": DQNAgent,
    "q_learning": QLearningAgent,
}

# Agorithms that do not require training (skip the training and straight to eval)
BASELINE_ALGOS = ["max_pressure", "random", "fixed_time"]

# Agent hyperparameters
PARAM_GRID = {
    "ppo": {
        "lr": [3e-4],  
        "gamma": [0.99],  
        "clip": [0.2],
        "gae_lambda": [0.95], 
        "K": [10],  
        "entropy_coef": [0.01, 0.02],  
    },
    "max_pressure": {
        "ped_wait_weight": [1.0,]
    },
    "dqn": {
        "lr": [1e-3],
        "gamma": [0.95],
        "epsilon": [0.05],
        "batch_size": [64],
        "target_update_freq": [10],
    },
    "q_learning": {
        "lr": [0.1],
        "gamma": [0.95],
        "epsilon": [1.0],
        "eps_decay": [0.995],
        "eps_min": [0.01],
        "bins": [8],
    },
    "random": {
    },
    "fixed_time": {
    }
}

# """Redifinition with minimal config (just for internal testing without deleting the 
# above ones). Comment all the ones you dont want to test and keep the one you need."""
# ALGORITHMS = {
#     "max_pressure": MaxPressureAgent,
    
# }
# PARAM_GRID = {
#     "max_pressure": {
#         "ped_wait_weight": [1]
#     },
# }

# ALGORITHMS = {
#     "random": RandomAgent,
# }
# PARAM_GRID = {
#     "random": {
#     }
# }

# ALGORITHMS = {
#     "fixed_time": FixedTimeAgent,
# }
# PARAM_GRID = {
#     "fixed_time": {
#     },
# }

# ALGORITHMS = {
#     "ppo": PPOAgent,
# }
# PARAM_GRID = {
#     "ppo": {
#     }
# }

# ALGORITHMS = {
#     "dqn": DQNAgent,
# }
# PARAM_GRID = {
#     "dqn": {
#     }
# }

# Training parameters
TRAINING_CONFIG = {
    "train_episodes": NUM_EPISODES,
    "log_interval": 1,
    "eval_episodes": 10
}

# Where to store the results
RESULTS_ROOT = Path("Results")

def get_file_date():
    """Function to print the date"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(data:dict, fname:Path):
    """Utility function to save json"""
    with open(fname, "w") as f:
        json.dump(data, f, indent=4)

def init_env(algo:BaseAlgorithm, route_file_path:Path, sumo_seed:Optional[int]=None):
    kwargs = {
        "use_gui": False,
        "route_file": str(route_file_path),
        "fixed_ts": isinstance(algo, FixedTimeAgent)
    }
    if sumo_seed is not None:
        kwargs["sumo_seed"] = sumo_seed
    return gym.make(CUSTOM_ENV_ID, **kwargs)

def train_algorithm(algo: BaseAlgorithm, training_config:dict, save_dir: Path):
    """Loop to train the algorithm using the training config dict. Iterates over all of 
    the 50 asymmetric route files"""
    results = []

    total_train_start = time.time()

    # Train the algorithm for the number of training episodes
    for episode in range(training_config["train_episodes"]):
        
        # Set the route file when creating the env, update the env in the algo object
        route_file_path = get_route_file_name(ASSYMETRIC_ROUTES_DIR, episode)
        env = init_env(algo, route_file_path, SUMO_SEED)
        algo.set_env(env)
        
        # Start the timer
        episode_start = time.time()
        
        # Reset the environment and flags
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = algo.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            #print("Next obs:", next_obs)
            algo.train_step((obs, action, reward, next_obs, done or truncated))
            obs = next_obs
            total_reward += reward

        # Reset the algorithm at teh end of the episode
        algo.reset()
        
        # Properly close the env and garbage-collect it
        env.close()
        del env
        
        episode_time = time.time() - episode_start
        results.append({
            "episode": episode,
            "reward": total_reward,
            "duration_sec": episode_time
        })
        
        if episode % training_config.get("log_interval", 1) == 0:
            print(f"Episode {episode} reward: {total_reward:.2f} | "
                  f"time: {episode_time:.2f}s")

    total_train_time = time.time() - total_train_start
    print(f"Total training time: {total_train_time:.2f}s")
    
    # Save the current algorithm state
    algo.save(save_dir / "algo.checkpoint")

    return {
        "episode_metrics": results,
        "total_training_time_sec": total_train_time
    }


def evaluate_algorithm(algo:BaseAlgorithm, route_file_indices:np.ndarray):
    """Function to evaluate the algorithm performance. This function loads the route 
    files at the given route_file_indices such that all algorithms are tested on the 
    same route files."""
    
    rewards = []
    route_indices = []
    episode_times = []
    
    eval_start = time.time()

    # Evaluate the algorithm for the number of episodes
    for index, route_file_index in enumerate(route_file_indices):
        ep_start = time.time()
        
        # Set the route file when creating the env, update the env in the algo object
        route_file_path = get_route_file_name(ASSYMETRIC_ROUTES_DIR, route_file_index)
        env = init_env(algo, route_file_path, SUMO_SEED)
        algo.set_env(env)
        
        obs, _ = env.reset()
        algo.reset()
        total = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = algo.select_action(obs, training=False)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward

        rewards.append(total)
        route_indices.append(index)
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        
        # Properly close the env and garbage-collect it
        env.close()
        del env
        
        print(f"Episode {index} reward: {total:.2f} | "
                  f"time: {ep_time:.2f}s")
        
    total_eval_time = time.time() - eval_start
    print(f"Finished evaluation in {total_eval_time} seconds")
    
    # Return the result metrics
    return {
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "all_rewards": rewards,
        "per_episode_time_sec": episode_times,
        "total_eval_time_sec": total_eval_time,
        "route_indices": route_file_indices.tolist(),
    }

def run(
    algorithms:dict=ALGORITHMS, 
    hyperparams:dict=PARAM_GRID, 
    training_config:dict=TRAINING_CONFIG, 
    results_root:Path=RESULTS_ROOT,
    baseline_algos:list[str]=BASELINE_ALGOS,
    random_seed:int=RANDOM_SEED,
):
    """Function to iterate over all combinations of algorithms and hyperparameters to 
    train and evaluate algorithms at the TSC task. Saves the algorithm end states, 
    training and eval results. This funciton essentially performs grid search.
    """

    # Create specific results directory under Results
    base_dir = Path(results_root) / get_file_date()
    base_dir.mkdir(parents=True, exist_ok=True)

    """Generate a list of route file indices to use in evaluation (each algo gets the 
    same files for evaluation)"""
    np.random.seed(random_seed)
    route_file_indices = np.random.choice(
        training_config["train_episodes"], 
        training_config["eval_episodes"], 
        replace=False
    ) 

    """Create a temp env to extract the num_obs and num_actions to init algos with 
    because all algos have these two parameters in their constructor (regardless if 
    they use them or not, and because all route files share the same number of obs and 
    actions)"""
    temp_env = gym.make(CUSTOM_ENV_ID)
    num_obs = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    temp_env.close()

    for algo_name, algo_class in algorithms.items():
        
        # Get the hyperparameter lists
        params_keys = list(hyperparams[algo_name].keys())
        params_values = list(hyperparams[algo_name].values())

        # Try all combinations of all parameters
        for params_tuple in product(*params_values):
                
            # Create the dict for keeping track of current config
            params_dict = dict(zip(params_keys, params_tuple))

            print(f"Running {algo_name} with {params_dict}")

            # Create folder Results/{timestamp}/algo_{name}_{k1}_{v1}_{k2}_{v2}
            name = algo_name + "_".join(f"{k}_{v}" for k, v in params_dict.items())
            save_dir = base_dir / name
            save_dir.mkdir(parents=True, exist_ok=True)

            # Initialize algorithm with the current iteration of its hyperparameters
            algo = algo_class(num_obs, num_actions, **params_dict)

            current_train_config = training_config.copy()
            if algo_name in baseline_algos:
                current_train_config["train_episodes"] = 0 
                print(f"Skipping training loop for {algo_name}")
                
            # Train and log the metrics (always close the env)
            train_metrics = train_algorithm(algo, current_train_config, save_dir)
            eval_metrics = evaluate_algorithm(algo, route_file_indices)
            
            # Save JSON logs
            save_json(train_metrics, save_dir / "train.json")
            save_json(eval_metrics, save_dir / "eval.json")


if __name__ == "__main__":
    run()
