from itertools import product
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from algorithms.base import BaseAlgorithm
from algorithms.PPO import PPO
from algorithms.q_learning import QLearningAgent
#from algorithms.MaxPressure import MaxPressure

import gymnasium as gym
from custom_env import CUSTOM_ENV_ID

# Reference for the algorithms evaluated
ALGORITHMS = {
    "ppo": PPO,
    #"max_pressure": MaxPressureAlgorithm,
    "q-learning": QLearningAgent,
}

# Hyperparameter grid
PARAM_GRID = {
    "ppo": {
        "lr": 3e-4,
        "gamma": 0.99,
        "clip": 0.2,
        "gae_lambda": 0.95,
        "K": 4,
    },
    "q-learning": {
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "eps_decay": 0.995,
        "eps_min": 0.01,
    },
    "max_pressure": {
    }
}

# Training parameters
TRAINING_CONFIG = {
    "train_steps": 500000,
    "log_interval": 1000,
    "eval_episodes": 10
}

# Where to store the results
RESULTS_ROOT = Path("Results")

def get_file_date():
    """Function to print the date"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def train_algorithm(env:gym.Env, algo: BaseAlgorithm, training_config:dict, save_dir: Path):
    """Loop to train the algorithm using the trasining config dict"""
    results = []
    obs, _ = env.reset()
    algo.reset()

    # Train the algorithm for the number of steps
    for step in range(training_config["train_steps"]):
        action = algo.select_action(obs)
        next_obs, reward, done, truncated,_ = env.step(action)

        algo.train_step((obs, action, reward, next_obs, done or truncated))

        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

        if step % training_config["log_interval"] == 0:
            results.append({"step": step, "reward": float(reward)})

    # Save the current model state
    algo.save(save_dir / "model.zip")

    return results


def evaluate_algorithm(env:gym.Env, algo:BaseAlgorithm, config:dict):
    """Function to evaluate the algorithm performance"""
    rewards = []

    # Evaluate the algorithm for the number of episodes
    for _ in range(config["eval_episodes"]):
        obs, _ = env.reset()
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
    env = gym.make(CUSTOM_ENV_ID)

    # Create specific results directory under Results
    base_dir = results_root / get_file_date()
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

            # Create folder: Results/timestamp/algo_lr=...
            name = algo_name + "_" + "_".join(f"{k}_{v}" for k, v in params_dict.items())
            save_dir = base_dir / name
            save_dir.mkdir(parents=True, exist_ok=True)

            # Initialize algorithm with the current iteration of its hyperparameters
            algo = algo_class(env, **params_dict)

            # Train and log the metrics
            train_metrics = train_algorithm(env, algo, training_config, save_dir)
            eval_metrics = evaluate_algorithm(env, algo, training_config)

            # Save JSON logs
            with open(save_dir / "train.json", "w") as f:
                json.dump(train_metrics, f, indent=4)

            with open(save_dir / "eval.json", "w") as f:
                json.dump(eval_metrics, f, indent=4)


if __name__ == "__main__":
    run()
