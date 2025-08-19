#!/usr/bin/env python3
"""
Model Evaluation Script for RL4AVS

This script loads pre-trained models and evaluates them against each other
using key performance indicators (KPIs) like collision rate, completion time,
safety metrics, etc.

Usage:
    python evaluate_models.py --models path1.zip path2.zip path3.zip --episodes 100
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import wandb

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from stable_baselines3 import PPO, A2C, DQN
from src.environment import make_env
import gymnasium as gym


class ModelEvaluator:
    """Evaluates and compares pre-trained RL models"""
    
    def __init__(self, use_custom_env=True, render=False):
        self.use_custom_env = use_custom_env
        self.render = render
        self.results = {}
        
    def load_model(self, model_path):
        """Load a model from file path and detect algorithm type"""
        print(f"Loading model from: {model_path}")
        
        # Detect algorithm from path name
        if "PPO" in model_path or "ppo" in model_path:
            model = PPO.load(model_path)
            algo_name = "PPO"
        elif "A2C" in model_path or "a2c" in model_path:
            model = A2C.load(model_path)
            algo_name = "A2C"
        elif "DQN" in model_path or "dqn" in model_path:
            model = DQN.load(model_path)
            algo_name = "DQN"
        else:
            raise ValueError(f"Cannot detect algorithm type from path: {model_path}")
            
        print(f"✅ Successfully loaded {algo_name} model")
        return model, algo_name
    
    def evaluate_single_model(self, model, algo_name, n_episodes=100):
        """Evaluate a single model and return detailed metrics"""
        print(f"\n🔍 Evaluating {algo_name} model ({n_episodes} episodes)...")
        
        # Create environment
        env = make_env("roundabout-v0", custom=self.use_custom_env)
        if self.render:
            env = gym.make("roundabout-v0", render_mode="human")
        
        # Metrics collection
        episode_rewards = []
        episode_lengths = []
        collision_count = 0
        success_count = 0
        completion_times = []
        safety_achievement_times = []  # Time to reach 95% safety
        total_distance_traveled = []
        
        # Safety tracking
        consecutive_safe_episodes = 0
        safety_threshold = 0.95  # 95% success rate
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            episode_distance = 0
            collision_occurred = False
            
            # Track safety achievement
            current_success_rate = success_count / max(episode, 1)
            safety_achieved = False
            
            while True:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # Track distance if available in info
                if 'distance_traveled' in info:
                    episode_distance = info['distance_traveled']
                
                # Check for safety achievement during episode
                if not safety_achieved and current_success_rate >= safety_threshold:
                    safety_achievement_times.append(step_count)
                    safety_achieved = True
                
                if done or truncated:
                    break
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            total_distance_traveled.append(episode_distance)
            
            # Check success/collision
            if info.get('crashed', False) or info.get('collision', False):
                collision_count += 1
                collision_occurred = True
                consecutive_safe_episodes = 0
            else:
                success_count += 1
                consecutive_safe_episodes += 1
                completion_times.append(step_count)
            
            # Progress indicator
            if (episode + 1) % 20 == 0:
                current_success_rate = success_count / (episode + 1)
                current_collision_rate = collision_count / (episode + 1)
                print(f"  Episode {episode + 1}/{n_episodes} - Success: {current_success_rate:.2%}, Collision: {current_collision_rate:.2%}")
        
        evaluation_time = time.time() - start_time
        
        # Calculate final metrics
        success_rate = success_count / n_episodes
        collision_rate = collision_count / n_episodes
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)
        mean_completion_time = np.mean(completion_times) if completion_times else None
        mean_safety_achievement_time = np.mean(safety_achievement_times) if safety_achievement_times else None
        
        # Efficiency metrics
        reward_per_step = mean_reward / mean_episode_length if mean_episode_length > 0 else 0
        
        env.close()
        
        metrics = {
            'algorithm': algo_name,
            'n_episodes': n_episodes,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_episode_length': mean_episode_length,
            'mean_completion_time': mean_completion_time,
            'mean_safety_achievement_time': mean_safety_achievement_time,
            'reward_per_step': reward_per_step,
            'evaluation_time_seconds': evaluation_time,
            'consecutive_safe_episodes_max': consecutive_safe_episodes,
            'total_successful_completions': success_count,
            'total_collisions': collision_count,
            'mean_distance_traveled': np.mean(total_distance_traveled) if total_distance_traveled else 0,
            'raw_episode_rewards': episode_rewards,
            'raw_episode_lengths': episode_lengths,
            'raw_completion_times': completion_times,
            'raw_safety_achievement_times': safety_achievement_times
        }
        
        print(f"✅ {algo_name} evaluation complete:")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Collision Rate: {collision_rate:.2%}")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Mean Episode Length: {mean_episode_length:.1f} steps")
        if mean_completion_time:
            print(f"   Mean Completion Time: {mean_completion_time:.1f} steps")
        if mean_safety_achievement_time:
            print(f"   Mean Time to 95% Safety: {mean_safety_achievement_time:.1f} steps")
        
        return metrics
    
    def compare_models(self, model_paths, n_episodes=100):
        """Compare multiple models and generate comprehensive report"""
        print("🚀 Starting Model Comparison Evaluation")
        print("=" * 60)
        
        # Initialize WandB for logging
        wandb.init(
            project="rl4avs_model_comparison",
            mode="offline",  # Run in offline mode
            config={
                "n_episodes": n_episodes,
                "use_custom_env": self.use_custom_env,
                "evaluation_timestamp": datetime.now().isoformat(),
                "models_evaluated": len(model_paths)
            }
        )
        
        all_results = {}
        
        for model_path in model_paths:
            try:
                # Load model
                model, algo_name = self.load_model(model_path)
                
                # Evaluate model
                metrics = self.evaluate_single_model(model, algo_name, n_episodes)
                all_results[algo_name] = metrics
                
                # Log to WandB
                wandb.log({
                    f"{algo_name}/success_rate": metrics['success_rate'],
                    f"{algo_name}/collision_rate": metrics['collision_rate'],
                    f"{algo_name}/mean_reward": metrics['mean_reward'],
                    f"{algo_name}/mean_episode_length": metrics['mean_episode_length'],
                    f"{algo_name}/reward_per_step": metrics['reward_per_step'],
                    f"{algo_name}/evaluation_time": metrics['evaluation_time_seconds'],
                })
                
                if metrics['mean_completion_time']:
                    wandb.log({f"{algo_name}/mean_completion_time": metrics['mean_completion_time']})
                if metrics['mean_safety_achievement_time']:
                    wandb.log({f"{algo_name}/mean_safety_achievement_time": metrics['mean_safety_achievement_time']})
                
            except Exception as e:
                print(f"❌ Error evaluating {model_path}: {e}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        # Save results
        self.save_results(all_results)
        
        wandb.finish()
        return all_results
    
    def generate_comparison_report(self, results):
        """Generate and print a detailed comparison report"""
        print("\n" + "=" * 60)
        print("📊 MODEL COMPARISON REPORT")
        print("=" * 60)
        
        if not results:
            print("❌ No models successfully evaluated")
            return
        
        # Create comparison table
        df_data = []
        for algo, metrics in results.items():
            df_data.append({
                'Algorithm': algo,
                'Success Rate': f"{metrics['success_rate']:.2%}",
                'Collision Rate': f"{metrics['collision_rate']:.2%}",
                'Mean Reward': f"{metrics['mean_reward']:.2f}",
                'Episode Length': f"{metrics['mean_episode_length']:.1f}",
                'Completion Time': f"{metrics['mean_completion_time']:.1f}" if metrics['mean_completion_time'] else "N/A",
                'Safety Time': f"{metrics['mean_safety_achievement_time']:.1f}" if metrics['mean_safety_achievement_time'] else "N/A",
                'Efficiency': f"{metrics['reward_per_step']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Find best performers
        print("\n🏆 BEST PERFORMERS:")
        
        # Best safety (lowest collision rate)
        best_safety = min(results.items(), key=lambda x: x[1]['collision_rate'])
        print(f"   🛡️  Safest Model: {best_safety[0]} ({best_safety[1]['collision_rate']:.2%} collision rate)")
        
        # Best success rate
        best_success = max(results.items(), key=lambda x: x[1]['success_rate'])
        print(f"   ✅ Highest Success: {best_success[0]} ({best_success[1]['success_rate']:.2%} success rate)")
        
        # Best efficiency
        best_efficiency = max(results.items(), key=lambda x: x[1]['reward_per_step'])
        print(f"   ⚡ Most Efficient: {best_efficiency[0]} ({best_efficiency[1]['reward_per_step']:.3f} reward/step)")
        
        # Fastest completion
        completion_times = {k: v['mean_completion_time'] for k, v in results.items() if v['mean_completion_time']}
        if completion_times:
            fastest = min(completion_times.items(), key=lambda x: x[1])
            print(f"   🏃 Fastest Completion: {fastest[0]} ({fastest[1]:.1f} steps average)")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   • For safety-critical applications, use the model with lowest collision rate")
        print("   • For general performance, consider the model with highest success rate")
        print("   • For efficiency, use the model with best reward-per-step ratio")
        
    def save_results(self, results):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = "experiments/test/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary CSV
        summary_data = []
        for algo, metrics in results.items():
            summary_data.append({
                'Algorithm': algo,
                'Success_Rate': metrics['success_rate'],
                'Collision_Rate': metrics['collision_rate'], 
                'Mean_Reward': metrics['mean_reward'],
                'Std_Reward': metrics['std_reward'],
                'Mean_Episode_Length': metrics['mean_episode_length'],
                'Mean_Completion_Time': metrics.get('mean_completion_time', None),
                'Mean_Safety_Achievement_Time': metrics.get('mean_safety_achievement_time', None),
                'Reward_Per_Step': metrics['reward_per_step'],
                'Evaluation_Time': metrics['evaluation_time_seconds']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{results_dir}/model_comparison_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
        
        # Save detailed results as JSON
        import json
        detailed_path = f"{results_dir}/detailed_results_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for algo, metrics in results.items():
                json_metrics = metrics.copy()
                for key, value in json_metrics.items():
                    if isinstance(value, np.ndarray):
                        json_metrics[key] = value.tolist()
                json_results[algo] = json_metrics
            json.dump(json_results, f, indent=2)
        print(f"💾 Detailed results saved to: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained RL models")
    parser.add_argument("--models", nargs='+', required=True,
                        help="Paths to model files to evaluate")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to evaluate each model")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes (will be slower)")
    parser.add_argument("--custom_env", action="store_true", default=True,
                        help="Use custom environment with enhanced rewards")
    
    args = parser.parse_args()
    
    print("🎯 RL4AVS Model Evaluation Framework")
    print("=" * 60)
    print(f"Models to evaluate: {len(args.models)}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Custom environment: {args.custom_env}")
    print(f"Render mode: {args.render}")
    print("=" * 60)
    
    # Verify model files exist
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
    
    # Create evaluator
    evaluator = ModelEvaluator(use_custom_env=args.custom_env, render=args.render)
    
    # Run evaluation
    results = evaluator.compare_models(args.models, args.episodes)
    
    print("\n🎉 Evaluation completed successfully!")
    print("Check experiments/test/results/ for detailed output files.")


if __name__ == "__main__":
    main()
