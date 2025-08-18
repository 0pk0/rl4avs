import json
import os
import pandas as pd
from datetime import datetime


def save_results(results, filename):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def create_tensorboard_logs(base_dir="results/logs"):
    """Create tensorboard log directory structure"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def print_experiment_summary(results):
    """Print a nice summary of experiment results"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for algo, algo_results in results.items():
        if isinstance(algo_results, list):
            rewards = [r['mean_reward'] for r in algo_results]
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum([(r - mean_reward) ** 2 for r in rewards]) / len(rewards)) ** 0.5
            print(f"{algo:>8}: {mean_reward:6.2f} ± {std_reward:5.2f} (n={len(rewards)} seeds)")
        else:
            print(f"{algo:>8}: {algo_results['mean_reward']:6.2f} ± {algo_results['std_reward']:5.2f}")


def create_readme():
    """Create a README file for the project"""
    readme_content = """# Highway-Env Roundabout RL Experiments

This project implements and compares different reinforcement learning algorithms on the highway-env roundabout environment.

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run training: `python experiments/train_ppo.py`
3. Run comparison: `python experiments/compare_algorithms.py`

## Structure
- `src/`: Core implementation files
- `experiments/`: Main experiment scripts
- `results/`: Saved models, logs, and plots
- `notebooks/`: Jupyter notebooks for analysis

## Results
Check `results/logs/` for training results and `results/plots/` for visualizations.
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("README.md created!")
