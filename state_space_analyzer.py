# state_space_analyzer.py
"""
State Space Analysis Tool for Q-learning
Author: Gator Guo
"""

import numpy as np

def analyze_state_space(n_features, n_actions):
    """
    Analyze Q-table requirements for different discretization strategies.
    
    Args:
        n_features: number of state features
        n_actions: number of possible actions
    """
    print("=" * 60)
    print("Q-LEARNING STATE SPACE ANALYSIS")
    print("=" * 60)
    print(f"State Features: {n_features}")
    print(f"Actions: {n_actions}\n")
    
    print("Q-TABLE SIZE ESTIMATES:")
    for bins in [5, 10, 15, 20]:
        n_states = bins ** n_features
        qtable_size = n_states * n_actions
        memory_mb = qtable_size * 8 / (1024**2)
        
        status = "✓ Good" if qtable_size < 1_000_000 else "⚠ Large"
        print(f"  {bins} bins/feature: {n_states:,} states, "
              f"{qtable_size:,} Q-values, {memory_mb:.1f}MB [{status}]")
    
    print("\nRECOMMENDATIONS:")
    print("  - Start with 10 bins per feature")
    print("  - If Q-table is too large, reduce bins or use fewer features")
    print("  - Monitor convergence during training")
    print("=" * 60)

if __name__ == "__main__":
    # Example: 8 features, 4 actions (typical traffic signal)
    analyze_state_space(n_features=8, n_actions=4)