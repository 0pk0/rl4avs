import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from .environment import make_env
from .utils import save_results


class DetailedEvaluator:
    def __init__(self, env_name="roundabout-v0", use_custom_env=False):
        self.env_name = env_name
        self.use_custom_env = use_custom_env
        self.metrics = defaultdict(list)

    def evaluate_agent(self, model, n_episodes=100, render=False):
        """Evaluate a single agent with detailed metrics"""
        env = make_env(self.env_name, custom=self.use_custom_env)
        if render:
            env = gym.make(self.env_name, render_mode="human")

        episode_rewards = []
        collision_count = 0
        success_count = 0
        episode_lengths = []
        comfort_metrics = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            prev_speed = 0
            acceleration_variance = []

            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                # Calculate comfort metrics (acceleration variance)
                if hasattr(obs, 'shape') and len(obs.shape) > 1 and obs.shape[1] > 4:
                    current_speed = np.sqrt(obs[0, 3] ** 2 + obs[0, 4] ** 2)
                    acceleration = current_speed - prev_speed
                    acceleration_variance.append(acceleration ** 2)
                    prev_speed = current_speed

            # Analyze episode outcome
            if info.get('crashed', False):
                collision_count += 1
            elif info.get('arrived', False):
                success_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            comfort_metrics.append(np.mean(acceleration_variance) if acceleration_variance else 0)

        # Compile results
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'collision_rate': collision_count / n_episodes,
            'success_rate': success_count / n_episodes,
            'mean_episode_length': np.mean(episode_lengths),
            'comfort_score': np.mean(comfort_metrics),
            'episode_rewards': episode_rewards,
            'n_episodes': n_episodes
        }

        env.close()
        return results

    def compare_algorithms(self, model_paths, n_episodes=50):
        """Compare multiple algorithms"""
        comparison_results = {}

        for algo_name, model_path in model_paths.items():
            print(f"Evaluating {algo_name}...")
            model = self.load_model(algo_name, model_path)
            results = self.evaluate_agent(model, n_episodes=n_episodes)
            comparison_results[algo_name] = results

        return comparison_results

    def load_model(self, algo_name, model_path):
        """Load a trained model"""
        if algo_name == 'DQN':
            return DQN.load(model_path)
        elif algo_name == 'PPO':
            return PPO.load(model_path)
        elif algo_name == 'A2C':
            return A2C.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    def create_comparison_plots(self, results, save_path="results/plots/algorithm_comparison.png"):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        algorithms = list(results.keys())

        # Reward comparison
        rewards = [results[algo]['mean_reward'] for algo in algorithms]
        reward_stds = [results[algo]['std_reward'] for algo in algorithms]
        axes[0, 0].bar(algorithms, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Mean Episode Reward', fontsize=14)
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Success rate comparison
        success_rates = [results[algo]['success_rate'] for algo in algorithms]
        axes[0, 1].bar(algorithms, success_rates, alpha=0.7, color='green')
        axes[0, 1].set_title('Success Rate', fontsize=14)
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # Collision rate comparison
        collision_rates = [results[algo]['collision_rate'] for algo in algorithms]
        axes[1, 0].bar(algorithms, collision_rates, alpha=0.7, color='red')
        axes[1, 0].set_title('Collision Rate', fontsize=14)
        axes[1, 0].set_ylabel('Collision Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        # Episode length comparison
        episode_lengths = [results[algo]['mean_episode_length'] for algo in algorithms]
        axes[1, 1].bar(algorithms, episode_lengths, alpha=0.7, color='blue')
        axes[1, 1].set_title('Mean Episode Length', fontsize=14)
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {save_path}")
        plt.show()

    def create_performance_table(self, results):
        """Create a detailed performance table"""
        performance_data = {}

        for algo, result in results.items():
            performance_data[algo] = {
                'Mean Reward': f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                'Success Rate': f"{result['success_rate']:.2%}",
                'Collision Rate': f"{result['collision_rate']:.2%}",
                'Avg Episode Length': f"{result['mean_episode_length']:.1f}",
                'Comfort Score': f"{result['comfort_score']:.4f}"
            }

        df = pd.DataFrame(performance_data).T
        return df
