"""For enhanced performance (using libsumo C++ backend), then call 
"export LIBSUMO_AS_TRACI=1" before launching this script.
 """


from itertools import product
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from algorithms.base import BaseAlgorithm
from algorithms.PPO import PPOAgent
from algorithms.q_learning import QLearningAgent
#from algorithms.MaxPressure import MaxPressure

import gymnasium as gym
from custom_env import CUSTOM_ENV_ID

# Reference for the algorithms evaluated
""" ALGORITHMS = {
    "ppo": PPOAgent,
    #"max_pressure": MaxPressureAlgorithm,
    "q-learning": QLearningAgent,
} """

ALGORITHMS = {
    "ppo": PPOAgent,
}

# Hyperparameter grid (each param has a list of candidate values)
""" PARAM_GRID = {
    "ppo": {
        "lr": [3e-4],
        "gamma": [0.99],
        "clip": [0.2],
        "gae_lambda": [0.95],
        "K": [4],
    },
    "q-learning": {
        "lr": [0.1],
        "gamma": [0.99],
        "epsilon": [1.0],
        "eps_decay": [0.995],
        "eps_min": [0.01],
    },
    "max_pressure": {
    }
} """

PARAM_GRID = {
    "ppo": {
        "lr": [3e-4],
        "gamma": [0.99],
        "clip": [0.2],
        "gae_lambda": [0.95],
        "K": [4],
    },
}


# Training parameters
TRAINING_CONFIG = {
    "train_episodes": 100,
    "log_interval": 10,
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

    print("Observation:", obs)
    print("Type:", type(obs))
    print("Dtype:", obs.dtype if isinstance(obs, np.ndarray) else None)
    print("Shape:", obs.shape)
    print("Min/Max:", np.min(obs), np.max(obs))

    # Train the algorithm for the number of training episodes
    for episode in range(training_config["train_episodes"]):
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

        if episode % training_config.get("log_interval", 1) == 0:
            print(f"Episode {episode+1} reward: {total_reward}")

    # Save the current algorithm state
    algo.save(save_dir / "algo.npz")

    return results


def evaluate_algorithm(env:gym.Env, algo:BaseAlgorithm, config:dict):
    """Function to evaluate the algorithm performance"""
    rewards = []

    # Evaluate the algorithm for the number of episodes
    for _ in range(config["eval_episodes"]):
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

    # Return the result metrics
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "all_rewards": rewards
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
            save_json(train_metrics, save_dir / "train.json", "w")
            save_json(eval_metrics, save_dir / "eval.json", "w")


if __name__ == "__main__":
    run()
