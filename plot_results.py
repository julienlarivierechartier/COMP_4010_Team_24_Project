import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_ROOT = Path("Results")

# RL Algorithms (Agents that learn and have training curves)
RL_ALGORITHMS = ["ppo", "dqn", "q_learning"] 

# Baseline Algorithms (Heuristics or static agents)
BASELINE_ALGORITHMS = ["max_pressure", "fixed_time", "random"]

# Smoothing window for training curves
SMOOTHING_WINDOW = 5

# Color Scheme for Consistency across graphs
ALGO_COLORS = {
    # RL
    'ppo': 'blue',
    'dqn': 'green',
    'q_learning': 'purple',
    
    # Baselines
    'max_pressure': 'red',
    'fixed_time': 'orange',
    'random': 'gray'
}

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def get_latest_date_folder(root_path):
    """Finds the most recent timestamp folder in Results"""
    if not root_path.exists():
        return None
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs, key=lambda x: x.name)[-1]

def moving_average(values, window):
    """Smooths the data array"""
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    # Valid mode 'same' prevents size mismatch
    return np.convolve(values, weights, 'same')

def find_best_folder(date_folder, algo_prefix, is_rl=True):
    """
    Finds the specific sub-folder for an algorithm that performed best.
    - RL: Best average reward in the last 3 training episodes.
    - Baselines: Best average reward in the evaluation file.
    """
    best_score = -float('inf')
    best_folder = None
    best_name = None

    print(f"Searching best configuration for {algo_prefix}...")

    for folder in date_folder.iterdir():
        if not folder.is_dir() or not folder.name.startswith(algo_prefix):
            continue

        try:
            current_score = -float('inf')

            if is_rl:
                # For RL, we pick the one that finished training the strongest
                train_file = folder / "train.json"
                if train_file.exists():
                    with open(train_file, 'r') as f:
                        data = json.load(f)
                    metrics = data.get("episode_metrics", [])
                    if metrics:
                        rewards = [m["reward"] for m in metrics]
                        # Score is avg of last 3 episodes
                        current_score = np.mean(rewards[-3:]) if len(rewards) >= 3 else np.mean(rewards)
            else:
                # For Baselines, we pick the one with best eval average
                eval_file = folder / "eval.json"
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                    current_score = data.get("avg_reward", -float('inf'))

            if current_score > best_score:
                best_score = current_score
                best_folder = folder
                best_name = folder.name
        
        except Exception as e:
            # print(f"  Error reading {folder.name}: {e}")
            pass

    if best_folder:
        print(f"  -> Best {algo_prefix}: {best_name} (Score: {best_score:.2f})")
    else:
        print(f"  -> No data found for {algo_prefix}")

    return best_folder

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_training_only(best_folders):
    """GRAPH 1: Training Curves (Only for RL Algorithms)"""
    plt.figure(figsize=(10, 6))
    has_data = False

    for algo, folder in best_folders.items():
        if algo not in RL_ALGORITHMS or folder is None:
            continue
            
        train_file = folder / "train.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
            
            metrics = data.get("episode_metrics", [])
            x = [m["episode"] for m in metrics]
            y = [m["reward"] for m in metrics]
            
            y_smooth = moving_average(y, SMOOTHING_WINDOW)
            color = ALGO_COLORS.get(algo, 'black')
            
            plt.plot(x, y_smooth, linewidth=2, label=f"{algo}", color=color)
            plt.plot(x, y, linewidth=1, alpha=0.15, color=color) # Faint raw data
            has_data = True

    if has_data:
        plt.title("Training Curves: RL Agents Comparison")
        plt.xlabel("Training Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("graph_1_training_episodes.png", dpi=300)
        print("Saved: graph_1_training_episodes.png")
    else:
        print("Skipping Training Graph (No RL data found)")

def plot_eval_only(best_folders):
    """GRAPH 2: Evaluation Episodes (Line chart of the 10 test episodes)"""
    plt.figure(figsize=(10, 6))
    has_data = False

    # Combined list of all algos to iterate through
    all_algos = RL_ALGORITHMS + BASELINE_ALGORITHMS

    for algo in all_algos:
        folder = best_folders.get(algo)
        if folder is None:
            continue
            
        eval_file = folder / "eval.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            rewards = data.get("all_rewards", [])
            if rewards:
                episodes = range(1, len(rewards) + 1)
                
                color = ALGO_COLORS.get(algo, 'black')
                # Dashed line for baselines, solid for RL
                linestyle = '--' if algo in BASELINE_ALGORITHMS else '-'
                
                plt.plot(episodes, rewards, 
                         label=f"{algo} (Avg: {np.mean(rewards):.0f})", 
                         color=color, 
                         linestyle=linestyle,
                         marker='o', markersize=4, alpha=0.8)
                has_data = True

    if has_data:
        plt.title("Evaluation Episodes: Stability Check (All Algorithms)")
        plt.xlabel("Test Episode Index")
        plt.ylabel("Reward")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout() # Fix legend clipping
        plt.savefig("graph_2_eval_episodes.png", dpi=300)
        print("Saved: graph_2_eval_episodes.png")

def plot_combined_comparison(best_folders):
    """GRAPH 3: The Combined Graph (RL Training Curves vs Baseline Horizontal Lines)"""
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Baselines (Horizontal Lines)
    for algo in BASELINE_ALGORITHMS:
        folder = best_folders.get(algo)
        if folder:
            eval_file = folder / "eval.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    val = json.load(f).get("avg_reward")
                
                color = ALGO_COLORS.get(algo, 'gray')
                plt.axhline(y=val, color=color, linestyle='--', 
                            linewidth=2, label=f"{algo} (Avg: {val:.0f})")

    # 2. Plot RL Training Curves
    for algo in RL_ALGORITHMS:
        folder = best_folders.get(algo)
        if folder:
            train_file = folder / "train.json"
            if train_file.exists():
                with open(train_file, 'r') as f:
                    metrics = json.load(f).get("episode_metrics", [])
                
                x = [m["episode"] for m in metrics]
                y = [m["reward"] for m in metrics]
                y_smooth = moving_average(y, SMOOTHING_WINDOW)
                
                color = ALGO_COLORS.get(algo, 'black')
                plt.plot(x, y_smooth, linewidth=2.5, label=f"{algo} Curve", color=color)
                plt.plot(x, y, linewidth=0.5, alpha=0.15, color=color) # Faint raw

    plt.title("Master Comparison: RL Training vs Baseline Benchmarks")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("graph_3_combined_comparison.png", dpi=300)
    print("Saved: graph_3_combined_comparison.png")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    latest_date_folder = get_latest_date_folder(RESULTS_ROOT)
    
    if latest_date_folder:
        print(f"Processing results from: {latest_date_folder}\n")
        
        # 1. Find the best folder for each algorithm
        best_run_folders = {}
        
        # Check RL Algos
        for algo in RL_ALGORITHMS:
            best_run_folders[algo] = find_best_folder(latest_date_folder, algo, is_rl=True)
            
        # Check Baselines
        for algo in BASELINE_ALGORITHMS:
            best_run_folders[algo] = find_best_folder(latest_date_folder, algo, is_rl=False)
            
        print("\nGenerating Graphs...")
        
        # 2. Generate the 3 requested graphs
        plot_training_only(best_run_folders)
        plot_eval_only(best_run_folders)
        plot_combined_comparison(best_run_folders)
        
        plt.show() 
    else:
        print("No results directory found.")