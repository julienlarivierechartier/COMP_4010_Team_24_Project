import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_ROOT = Path("Results")

# Algorithms to include in the plot
RL_ALGORITHMS = ["ppo"]  # Agents that have training data
BASELINE_ALGORITHMS = ["max_pressure", "fixed_time"] # Heuristics

# Smoothing: Window size for moving average
SMOOTHING_WINDOW = 5

# ==========================================
# HELPER FUNCTIONS
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
    """Smooths the data array to make graphs look cleaner"""
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    # FIX: Used 'same' to keep dimensions consistent
    return np.convolve(values, weights, 'same')

def load_best_rl_run(date_folder, algo_prefix):
    """
    Scans ALL folders starting with 'algo_prefix'
    and returns the data for the ONE run that had the best final performance.
    """
    best_score = -float('inf')
    best_data = None
    best_name = None

    print(f"Scanning for best {algo_prefix} run...")

    for folder in date_folder.iterdir():
        if not folder.is_dir() or not folder.name.startswith(algo_prefix):
            continue

        train_file = folder / "train.json"
        if not train_file.exists():
            continue

        try:
            with open(train_file, 'r') as f:
                data = json.load(f)
                
            metrics = data.get("episode_metrics", [])
            if not metrics:
                continue

            rewards = [m["reward"] for m in metrics]
            episodes = [m["episode"] for m in metrics]

            # Calculate score based on the average of the last 3 episodes
            if len(rewards) >= 3:
                final_score = np.mean(rewards[-3:])
            else:
                final_score = np.mean(rewards)

            if final_score > best_score:
                best_score = final_score
                best_data = {"episodes": episodes, "rewards": rewards}
                best_name = folder.name
                
        except Exception as e:
            print(f"Skipping {folder.name}: {e}")

    if best_name:
        print(f"  -> Winner: {best_name} (Score: {best_score:.2f})")
    
    return best_name, best_data

def load_baseline_score(date_folder, algo_prefix):
    """
    Scans folders for baseline algos and finds the best eval score.
    """
    best_avg = -float('inf')
    
    for folder in date_folder.iterdir():
        if not folder.is_dir() or not folder.name.startswith(algo_prefix):
            continue

        eval_file = folder / "eval.json"
        if not eval_file.exists():
            continue

        with open(eval_file, 'r') as f:
            data = json.load(f)
            avg = data.get("avg_reward", -float('inf'))
            
            if avg > best_avg:
                best_avg = avg
                
    return best_avg if best_avg != -float('inf') else None

# ==========================================
# PLOTTING
# ==========================================

def plot_comparison(date_folder):
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Baselines as Horizontal Lines
    colors = {'max_pressure': 'red', 'fixed_time': 'orange'}
    
    for algo in BASELINE_ALGORITHMS:
        score = load_baseline_score(date_folder, algo)
        if score is not None:
            plt.axhline(y=score, color=colors.get(algo, 'gray'), linestyle='--', linewidth=2, label=f"{algo} (Baseline)")
            print(f"Plotting {algo} baseline at {score}")

    # 2. Plot Best RL Agent Curve
    for algo in RL_ALGORITHMS:
        name, data = load_best_rl_run(date_folder, algo)
        if data:
            x = data["episodes"]
            y = data["rewards"]
            
            # Smooth the line
            y_smooth = moving_average(y, SMOOTHING_WINDOW)
            
            # Ensure x and y lengths match after smoothing
            min_len = min(len(x), len(y_smooth))
            x = x[:min_len]
            y_smooth = y_smooth[:min_len]
            
            plt.plot(x, y_smooth, linewidth=2.5, label=f"PPO (Best Run)")
            
            # Plot the real data faintly behind it
            plt.plot(x, y[:min_len], linewidth=1, alpha=0.2, color='blue')

    plt.title("Training Progress: RL vs Baselines")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "final_result_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nGraph saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    latest = get_latest_date_folder(RESULTS_ROOT)
    if latest:
        print(f"Processing folder: {latest}")
        plot_comparison(latest)
    else:
        print("No results found.")