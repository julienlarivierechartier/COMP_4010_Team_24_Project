"""
Quick Q-Learning Model Checker
Name: Gator Guo
ID:101267370

Simple utility to check basic stats of trained Q-learning models.
"""

import numpy as np
from pathlib import Path


def check_model(model_path):
    """Load a Q-learning model and print basic info"""
    
    print(f"\n{'='*50}")
    print(f"Checking: {Path(model_path).name}")
    print(f"{'='*50}")
    
    # Load the model
    data = np.load(model_path, allow_pickle=True)
    
    # Get Q-table
    raw_table = data["q_table"]
    q_table = {tuple(state): values for state, values in raw_table}
    
    # Calculate stats
    bins = data["bins_per_feature"]
    max_states = np.prod(bins)
    visited = len(q_table)
    coverage = 100 * visited / max_states
    
    # Print results
    print(f"States visited:     {visited:,}")
    print(f"Max possible:       {max_states:,}")
    print(f"Coverage:           {coverage:.2f}%")
    print(f"Bins per feature:   {list(bins)}")
    print(f"Epsilon:            {data['epsilon']:.4f}")
    print(f"Learning rate:      {data['lr']:.4f}")
    print(f"Gamma:              {data['gamma']:.4f}")
    
    # Show Q-value range
    all_q = []
    for values in q_table.values():
        all_q.extend(values)
    
    if all_q:
        print(f"Q-value range:      [{min(all_q):.2f}, {max(all_q):.2f}]")
        print(f"Mean Q-value:       {np.mean(all_q):.2f}")
    
    print(f"{'='*50}\n")
    
    return {
        'visited': visited,
        'coverage': coverage,
        'epsilon': float(data['epsilon'])
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check model from command line
        for model_path in sys.argv[1:]:
            check_model(model_path)
    else:
        print("Usage: python check_qlearning_model.py <model.npz>")
        print("\nExample:")
        print("  python check_qlearning_model.py results/qlearning_trained.npz")