import os
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from .environment import make_env
from .utils import save_results, create_tensorboard_logs


class ExperimentRunner:
    def __init__(self, env_name="roundabout-v0", use_custom_env=False):
        self.env_name = env_name
        self.use_custom_env = use_custom_env
        self.algorithms = {
            'DQN': DQN,
            'PPO': PPO,
            'A2C': A2C,
        }
        self.results = {}

        # Create results directories
        os.makedirs("results/models", exist_ok=True)
        os.makedirs("results/logs", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)

    def train_algorithm(self, algo_name, total_timesteps=50000, n_seeds=3, save_model=True):
        print(f"\nTraining {algo_name} for {total_timesteps} timesteps...")
        seed_results = []

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}")

            # Create fresh environment for each seed
            env = make_env(self.env_name, custom=self.use_custom_env)
            env.reset(seed=seed)

            # Set up tensorboard logging
            log_dir = f"results/logs/{algo_name}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

            # Initialize model with tensorboard logging
            model = self.algorithms[algo_name](
                'MlpPolicy',
                env,
                verbose=1,
                seed=seed,
                tensorboard_log=log_dir
            )

            # Train model
            model.learn(total_timesteps=total_timesteps)

            # Evaluate
            print(f"    Evaluating {algo_name} seed {seed}...")
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)

            result = {
                'seed': seed,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'total_timesteps': total_timesteps,
                'algorithm': algo_name,
                'timestamp': datetime.now().isoformat()
            }

            seed_results.append(result)

            # Save model
            if save_model:
                model_path = f"results/models/{algo_name}_seed_{seed}"
                model.save(model_path)
                print(f"    Model saved to {model_path}")

            env.close()

        self.results[algo_name] = seed_results

        # Save results to JSON file
        save_results(self.results, f"results/logs/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        return seed_results

    def train_all_algorithms(self, total_timesteps=50000, n_seeds=3):
        """Train all algorithms and save comprehensive results"""
        print("Starting comprehensive training of all algorithms...")

        for algo in self.algorithms.keys():
            self.train_algorithm(algo, total_timesteps, n_seeds)

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)

        for algo, results in self.results.items():
            mean_rewards = [r['mean_reward'] for r in results]
            overall_mean = np.mean(mean_rewards)
            overall_std = np.std(mean_rewards)
            print(f"{algo}: {overall_mean:.2f} ± {overall_std:.2f}")

        return self.results
