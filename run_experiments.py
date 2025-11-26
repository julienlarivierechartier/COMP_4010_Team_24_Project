"""For enhanced performance (using libsumo C++ backend), then call 
"export LIBSUMO_AS_TRACI=1" before launching this script.
 """


from itertools import product
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import time

# Import the algorithms (agents)
from algorithms import (
    BaseAlgorithm, 
    PPOAgent, 
    QLearningAgent, 
    MaxPressureAgent,
    RandomAgent,
    FixedTimeAgent,
)

import gymnasium as gym
from custom_env import CUSTOM_ENV_ID

# Reference for the algorithms evaluated
ALGORITHMS = {
    "ppo": PPOAgent,
    "max_pressure": MaxPressureAgent,
    "random": RandomAgent,
    "fixed_time": FixedTimeAgent,
}

# Agent hyperparameters
PARAM_GRID = {
    "ppo": {
        "lr": [1e-4, 3e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.995],
        "clip": [0.1, 0.2, 0.3],
        "gae_lambda": [0.9, 0.95, 0.98],
        "K": [3, 4, 5, 8],
    },
    "max_pressure": {
        "ped_wait_weight": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    },
    "random": {
    },
    "fixed_time": {
        "cycle_phases": [
            None,
            [0, 1, 5, 6],
            [0, 5],
        ],
        "phase_durations": [
            None,
            [2, 1, 2, 1],
            [3, 3],
            [4, 2],
        ],
    }
}

# Training parameters
TRAINING_CONFIG = {
    "train_episodes": 50,
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

def train_algorithm(env:gym.Env, algo: BaseAlgorithm, training_config:dict, save_dir: Path):
    """Loop to train the algorithm using the training config dict"""
    results = []
    obs, _ = env.reset()
    algo.reset()

    total_train_start = time.time()

    print("Observation:", obs)
    print("Type:", type(obs))
    print("Dtype:", obs.dtype if isinstance(obs, np.ndarray) else None)
    print("Shape:", obs.shape)
    print("Min/Max:", np.min(obs), np.max(obs))

    # Train the algorithm for the number of training episodes
    for episode in range(training_config["train_episodes"]):
        episode_start = time.time()
        
        obs, _ = env.reset()
        algo.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = algo.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            #print("Next obs:", next_obs)
            algo.train_step((obs, action, reward, next_obs, done or truncated))
            obs = next_obs
            total_reward += reward

        episode_time = time.time() - episode_start
        results.append({
            "episode": episode + 1,
            "reward": total_reward,
            "duration_sec": episode_time
        })
        
        if episode % training_config.get("log_interval", 1) == 0:
            print(f"Episode {episode+1} reward: {total_reward} | "
                  f"time: {episode_time:.2f}s")

    total_train_time = time.time() - total_train_start
    print(f"Total training time: {total_train_time:.2f}s")
    
    # Save the current algorithm state
    algo.save(save_dir / "algo.checkpoint")

    return {
        "episode_metrics": results,
        "total_training_time_sec": total_train_time
    }


def evaluate_algorithm(env:gym.Env, algo:BaseAlgorithm, config:dict):
    """Function to evaluate the algorithm performance"""
    rewards = []
    episode_times = []
    
    eval_start = time.time()

    # Evaluate the algorithm for the number of episodes
    for _ in range(config["eval_episodes"]):
        ep_start = time.time()
        
        obs, _ = env.reset()
        algo.reset()
        total = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = algo.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward

        rewards.append(total)
        
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        print(f"Finished episode in {ep_time} seconds")
        
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
        "total_eval_time_sec": total_eval_time
    }

def run(
    algorithms:dict=ALGORITHMS, 
    hyperparams:dict=PARAM_GRID, 
    training_config:dict=TRAINING_CONFIG, 
    results_root:Path=RESULTS_ROOT
):
    """Function to iterate over all combinations of algorithms and hyperparameters to 
    train and evaluate algorithms at the TSC task. Saves the algorithm end states, 
    training and eval results. This funciton essentially performs grid search.
    """
    env = gym.make(CUSTOM_ENV_ID, use_gui=False)

    # Create specific results directory under Results
    base_dir = Path(results_root) / get_file_date()
    base_dir.mkdir(parents=True, exist_ok=True)

    for algo_name, algo_class in algorithms.items():
        
        # Get the hyperparameter lists
        params_keys = list(hyperparams[algo_name].keys())
        params_values = list(hyperparams[algo_name].values())

        # Try all combinations of all parameters
        for params_tuple in product(*params_values):
            
            # Create the dict for keeping track of current config
            params_dict = dict(zip(params_keys, params_tuple))

            print(f"Running {algo_name} with {params_dict}")

            # Create folder: Results/timestamp/algo_lr_{k1}_{v1}_{k2}_{v2} etc..
            name = algo_name + "_" + "_".join(f"{k}_{v}" for k, v in params_dict.items())
            save_dir = base_dir / name
            save_dir.mkdir(parents=True, exist_ok=True)

            # Initialize algorithm with the current iteration of its hyperparameters
            algo = algo_class(env, **params_dict)

            # Train and log the metrics
            train_metrics = train_algorithm(env, algo, training_config, save_dir)
            eval_metrics = evaluate_algorithm(env, algo, training_config)

            # Save JSON logs
            save_json(train_metrics, save_dir / "train.json")
            save_json(eval_metrics, save_dir / "eval.json")


if __name__ == "__main__":
    run()
