import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import DetailedEvaluator
from src.utils import save_results
import glob
from datetime import datetime


def main():
    print("Comparing Trained RL Algorithms")
    print("==============================")

    # Initialize evaluator
    evaluator = DetailedEvaluator(env_name="roundabout-v0")

    # Find trained models (using the first seed of each algorithm)
    model_paths = {}
    for algo in ['DQN', 'PPO', 'A2C']:
        model_pattern = f"results/models/{algo}_seed_0*"
        matches = glob.glob(model_pattern)
        if matches:
            model_paths[algo] = matches[0].replace('.zip', '')
            print(f"Found {algo} model: {model_paths[algo]}")
        else:
            print(f"No trained model found for {algo}")

    if not model_paths:
        print("No trained models found! Please run train_agents.py first.")
        return

    # Compare algorithms
    print("\nEvaluating algorithms...")
    comparison_results = evaluator.compare_algorithms(model_paths, n_episodes=50)

    # Create performance table
    performance_table = evaluator.create_performance_table(comparison_results)
    print("\nPerformance Comparison:")
    print(performance_table)

    # Create comparison plots
    evaluator.create_comparison_plots(comparison_results)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(comparison_results, f"results/logs/comparison_results_{timestamp}.json")

    # Save performance table
    performance_table.to_csv(f"results/logs/performance_table_{timestamp}.csv")
    print(f"\nResults saved to results/logs/")


if __name__ == "__main__":
    main()
