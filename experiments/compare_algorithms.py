#!/usr/bin/env python3
"""
📊 ALGORITHM COMPARISON: PPO vs Q-LEARNING 📊

This script provides comprehensive comparison between PPO and Q-Learning
agents trained on the same roundabout environment. Perfect for MSc analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import json
import gymnasium as gym
from stable_baselines3 import PPO
from src.q_learning_agent import QLearningAgent
from src.environment import register_custom_env


def evaluate_algorithm(model, algo_type, env, n_episodes=50):
    """Evaluate a single algorithm"""
    
    episode_rewards = []
    success_count = 0
    collision_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = truncated = False
        
        while not (done or truncated):
            if algo_type == 'PPO':
                action, _ = model.predict(obs, deterministic=True)
            else:  # Q-Learning
                action = model.get_action(obs, training=False)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        if info.get('crashed', False):
            collision_count += 1
        elif info.get('arrived', False):
            success_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / n_episodes,
        'collision_rate': collision_count / n_episodes,
        'episode_rewards': episode_rewards
    }


def compare_algorithms():
    """Main comparison function"""
    
    print("📊 ALGORITHM COMPARISON: PPO vs Q-LEARNING")
    print("=" * 50)
    
    # Load models
    models = {}
    
    # Try to load PPO
    ppo_paths = [
        "experiments/results/models/PPO_debug_seed_0.zip",
        "results/models/PPO_debug_seed_0.zip"
    ]
    
    for path in ppo_paths:
        if os.path.exists(path):
            try:
                models['PPO'] = PPO.load(path)
                print(f"✅ Loaded PPO from: {path}")
                break
            except Exception as e:
                print(f"❌ Error loading PPO: {e}")
    
    # Try to load Q-Learning
    q_paths = [
        "experiments/results/models/Q_Learning_final_seed_0.pkl",
        "experiments/results/models/Q_Learning_best_seed_0.pkl"
    ]
    
    for path in q_paths:
        if os.path.exists(path):
            try:
                models['Q_Learning'] = QLearningAgent.load_agent(path)
                print(f"✅ Loaded Q-Learning from: {path}")
                break
            except Exception as e:
                print(f"❌ Error loading Q-Learning: {e}")
    
    if not models:
        print("❌ No models found! Train algorithms first.")
        return
    
    # Create evaluation environment
    register_custom_env()
    env = gym.make('custom-roundabout-v0')
    
    # Evaluate each algorithm
    results = {}
    n_episodes = 50
    
    for algo_name, model in models.items():
        print(f"\n🔍 Evaluating {algo_name} ({n_episodes} episodes)...")
        results[algo_name] = evaluate_algorithm(model, algo_name, env, n_episodes)
        
        metrics = results[algo_name]
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print(f"   Collision Rate: {metrics['collision_rate']:.1%}")
        print(f"   Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # Generate comparison table
    print(f"\n📊 COMPARISON SUMMARY")
    print("-" * 50)
    print(f"{'Algorithm':<12} {'Success':<8} {'Collision':<10} {'Reward':<10}")
    print("-" * 50)
    
    for algo_name, metrics in results.items():
        print(f"{algo_name:<12} {metrics['success_rate']:.1%}     "
              f"{metrics['collision_rate']:.1%}       "
              f"{metrics['mean_reward']:.2f}")
    
    # Simple winner analysis
    print(f"\n🏆 WINNERS:")
    if len(results) >= 2:
        best_success = max(results.keys(), key=lambda x: results[x]['success_rate'])
        lowest_collision = min(results.keys(), key=lambda x: results[x]['collision_rate'])
        highest_reward = max(results.keys(), key=lambda x: results[x]['mean_reward'])
        
        print(f"   Best Success Rate: {best_success} ({results[best_success]['success_rate']:.1%})")
        print(f"   Lowest Collision Rate: {lowest_collision} ({results[lowest_collision]['collision_rate']:.1%})")
        print(f"   Highest Reward: {highest_reward} ({results[highest_reward]['mean_reward']:.2f})")
    
    # Create simple plot
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = list(results.keys())
        metrics = ['success_rate', 'collision_rate', 'mean_reward']
        titles = ['Success Rate (%)', 'Collision Rate (%)', 'Mean Reward']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = [results[algo][metric] for algo in algorithms]
            if i < 2:  # Convert rates to percentages
                values = [v * 100 for v in values]
            
            bars = axes[i].bar(algorithms, values, alpha=0.8)
            axes[i].set_title(title)
            axes[i].set_ylabel(title)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        os.makedirs("experiments/results", exist_ok=True)
        plt.savefig("experiments/results/algorithm_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved: experiments/results/algorithm_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
    
    env.close()
    return results


if __name__ == "__main__":
    compare_algorithms()