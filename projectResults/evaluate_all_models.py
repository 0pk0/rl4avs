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
import pickle # Added for Q-Learning model loading
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from stable_baselines3 import PPO, A2C, DQN
from src.environment import make_env
import gymnasium as gym
from src.q_learning_agent import QLearningAgent # Added for Q-Learning model loading


class ModelEvaluator:
    """Evaluates and compares pre-trained RL models"""
    
    def __init__(self, use_custom_env=True, render=False):
        self.use_custom_env = use_custom_env
        self.render = render
        self.results = {}
        self.writer = None # Initialize TensorBoard writer
        
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
        elif "Q_Learning" in model_path or "q_learning" in model_path:
            model = QLearningAgent.load_agent(model_path) # Use load_agent to get the QLearningAgent instance
            algo_name = "Q_Learning"
        else:
            raise ValueError(f"Cannot detect algorithm type from path: {model_path}")
            
        print(f"✅ Successfully loaded {algo_name} model")
        return model, algo_name
    
    def evaluate_single_model(self, model, algo_name, n_episodes=100, writer=None):
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
        near_miss_count = 0 # Added for near-miss frequency
        action_history_all_episodes = defaultdict(int) # To accumulate action distribution over all episodes
        
        # List to store per-episode metrics for detailed TensorBoard logging
        per_episode_metrics = defaultdict(list)

        # Safety tracking
        consecutive_safe_episodes = 0
        safety_threshold = 0.95  # 95% success rate

        # For smoothing metrics
        reward_window = []
        success_window = []
        collision_window = []
        near_miss_window = []
        safety_margin_window = []
        comfort_score_window = []
        window_size = 10 # For rolling averages
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            episode_distance = 0
            # action_history_current_episode = defaultdict(int) # For per-episode action distribution
            
            # Track safety achievement
            current_success_rate = success_count / max(episode, 1)
            safety_achieved = False
            
            while True:
                # Get action from model
                if algo_name == "Q_Learning":
                    # For Q-learning, the observation needs to be discretized
                    action = model.get_action(obs, training=False) # Use get_action for QLearningAgent
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray) and action.size == 1: # Convert single-element array to scalar
                        action = action.item()
                
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # Track action distribution
                action_history_all_episodes[action] += 1 # Accumulate for overall distribution
                # action_history_current_episode[action] += 1 # For per-episode action distribution
                
                # Track distance if available in info
                if 'distance_traveled' in info:
                    episode_distance = info['distance_traveled']
                
                # Check for near-miss if available in info
                if info.get('near_miss', False): # Placeholder for near-miss detection
                    near_miss_count += 1

                # Collect safety_margin and comfort_score per step
                if 'safety_margin' in info: 
                    per_episode_metrics['step_safety_margin'].append(info['safety_margin'])
                if 'comfort_score' in info:
                    per_episode_metrics['step_comfort_score'].append(info['comfort_score'])

                if done or truncated:
                    break
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            total_distance_traveled.append(episode_distance)
            
            # Check success/collision
            is_collision = info.get('crashed', False) or info.get('collision', False)
            is_success = False
            if is_collision:
                collision_count += 1
                # collision_occurred = True # Not used
                consecutive_safe_episodes = 0
            else:
                success_count += 1
                consecutive_safe_episodes += 1
                if info.get('arrived', False):
                    is_success = True
                completion_times.append(step_count)

            # Update rolling windows
            reward_window.append(episode_reward)
            success_window.append(1 if is_success else 0)
            collision_window.append(1 if is_collision else 0)
            near_miss_window.append(1 if info.get('near_miss', False) else 0)
            safety_margin_window.append(np.mean(per_episode_metrics['step_safety_margin']) if per_episode_metrics['step_safety_margin'] else 0)
            comfort_score_window.append(np.mean(per_episode_metrics['step_comfort_score']) if per_episode_metrics['step_comfort_score'] else 0)

            if len(reward_window) > window_size:
                reward_window.pop(0)
                success_window.pop(0)
                collision_window.pop(0)
                near_miss_window.pop(0)
                safety_margin_window.pop(0)
                comfort_score_window.pop(0)

            # Store per-episode metrics for TensorBoard
            per_episode_metrics['algo_name'].append(algo_name)
            per_episode_metrics['episode'].append(episode)
            per_episode_metrics['episode_reward'].append(episode_reward)
            per_episode_metrics['episode_length'].append(step_count)
            per_episode_metrics['is_collision'].append(is_collision)
            per_episode_metrics['is_success'].append(is_success)
            per_episode_metrics['is_near_miss'].append(info.get('near_miss', False))
            per_episode_metrics['mean_safety_margin'].append(np.mean(per_episode_metrics['step_safety_margin']) if per_episode_metrics['step_safety_margin'] else 0)
            per_episode_metrics['mean_comfort_score'].append(np.mean(per_episode_metrics['step_comfort_score']) if per_episode_metrics['step_comfort_score'] else 0)
            per_episode_metrics['smoothed_reward'].append(np.mean(reward_window))
            per_episode_metrics['smoothed_success_rate'].append(np.mean(success_window))
            per_episode_metrics['smoothed_collision_rate'].append(np.mean(collision_window))
            per_episode_metrics['smoothed_near_miss_frequency'].append(np.mean(near_miss_window))
            per_episode_metrics['smoothed_safety_margin'].append(np.mean(safety_margin_window))
            per_episode_metrics['smoothed_comfort_score'].append(np.mean(comfort_score_window))

            
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
        
        # Performance consistency (std dev of episode rewards)
        performance_consistency = std_reward
        
        # Comfort score (placeholder, requires environment info)
        # Assuming 'comfort_score' is available in info for each step and we average it
        # comfort_scores_per_episode = [info.get('comfort_score', 0) for _ in range(n_episodes)] # Placeholder
        mean_comfort_score = np.mean(per_episode_metrics['mean_comfort_score']) if per_episode_metrics['mean_comfort_score'] else 0 # Use collected mean

        mean_safety_achievement_time = np.mean(safety_achievement_times) if safety_achievement_times else None
        
        # Efficiency metrics
        reward_per_step = mean_reward / mean_episode_length if mean_episode_length > 0 else 0
        
        env.close()
        
        # Convert action_history keys to int for JSON serialization
        action_distribution_for_json = {int(k): v for k, v in action_history_all_episodes.items()}

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
            'raw_safety_achievement_times': safety_achievement_times,
            'near_miss_frequency': near_miss_count / n_episodes, # Added near-miss frequency
            'safety_margin_analysis': np.mean(per_episode_metrics['mean_safety_margin']) if per_episode_metrics['mean_safety_margin'] else 0, # Use collected mean
            'performance_consistency': performance_consistency, # Added performance consistency
            'mean_comfort_score': mean_comfort_score, # Added mean comfort score
            'action_distribution': action_distribution_for_json # Added action distribution
        }

        # Log total action distribution as scalars at the end of evaluation (global_step=n_episodes)
        if writer and action_history_all_episodes:
            # Ensure keys are strings for add_scalars
            action_dist_scalars = {f'Action_{int(k)}': v for k, v in action_history_all_episodes.items()}
            writer.add_scalars(f'{algo_name}/Total_Action_Distribution', action_dist_scalars, n_episodes)
        
        print(f"✅ {algo_name} evaluation complete:")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Collision Rate: {collision_rate:.2%}")
        print(f"   Near-miss Frequency: {metrics['near_miss_frequency']:.2%}") # Print near-miss frequency
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Performance Consistency (Std Reward): {performance_consistency:.2f}") # Print performance consistency
        print(f"   Mean Comfort Score: {mean_comfort_score:.2f}") # Print mean comfort score
        print(f"   Mean Episode Length: {mean_episode_length:.1f} steps")
        if mean_completion_time:
            print(f"   Mean Completion Time: {mean_completion_time:.1f} steps")
        if mean_safety_achievement_time:
            print(f"   Mean Time to 95% Safety: {mean_safety_achievement_time:.1f} steps")
        
        return metrics, per_episode_metrics
    
    def compare_models(self, model_paths, n_episodes=100):
        """Compare multiple models and generate comprehensive report"""
        print("🚀 Starting Model Comparison Evaluation")
        print("=" * 60)
        
        # Initialize TensorBoard writer
        log_dir = os.path.join("projectResults", "tensorboard_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logs will be saved to: {log_dir}")
        
        all_results = {}
        all_per_episode_metrics = {} # Dictionary to store per-episode metrics for comparative logging
        
        for model_path in model_paths:
            try:
                # Load model
                model, algo_name = self.load_model(model_path)
                
                # Evaluate model
                metrics, per_episode_metrics = self.evaluate_single_model(model, algo_name, n_episodes, self.writer) # Pass writer
                all_results[algo_name] = metrics
                
                # Log final metrics to TensorBoard for comparison across models in one view
                self.writer.add_scalar(f'{algo_name}/Final/success_rate', metrics['success_rate'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/collision_rate', metrics['collision_rate'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/near_miss_frequency', metrics['near_miss_frequency'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/mean_reward', metrics['mean_reward'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/performance_consistency', metrics['performance_consistency'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/mean_comfort_score', metrics['mean_comfort_score'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/mean_episode_length', metrics['mean_episode_length'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/reward_per_step', metrics['reward_per_step'], 0)
                self.writer.add_scalar(f'{algo_name}/Final/evaluation_time', metrics['evaluation_time_seconds'], 0)
                if metrics['mean_completion_time']:
                    self.writer.add_scalar(f'{algo_name}/Final/mean_completion_time', metrics['mean_completion_time'], 0)
                if metrics['mean_safety_achievement_time']:
                    self.writer.add_scalar(f'{algo_name}/Final/mean_safety_achievement_time', metrics['mean_safety_achievement_time'], 0)
                
                # Collect per-episode metrics for later comparative logging
                all_per_episode_metrics[algo_name] = per_episode_metrics
                
            except Exception as e:
                print(f"❌ Error evaluating {model_path}: {e}")
                continue

        # --- Generate comparative TensorBoard graphs ---
        if self.writer and all_per_episode_metrics:
            # Determine the maximum number of episodes to iterate through
            max_episodes_logged = max([len(pm['episode']) for pm in all_per_episode_metrics.values()])

            for episode_idx in range(max_episodes_logged):
                rewards_to_log = {}
                lengths_to_log = {}
                success_rates_to_log = {}
                collision_rates_to_log = {}
                near_miss_freqs_to_log = {}
                safety_margins_to_log = {}
                comfort_scores_to_log = {}
                
                for algo_name, pm in all_per_episode_metrics.items():
                    if episode_idx < len(pm['episode']):
                        rewards_to_log[algo_name] = pm['smoothed_reward'][episode_idx]
                        lengths_to_log[algo_name] = pm['episode_length'][episode_idx] # Individual episode length
                        success_rates_to_log[algo_name] = pm['smoothed_success_rate'][episode_idx]
                        collision_rates_to_log[algo_name] = pm['smoothed_collision_rate'][episode_idx]
                        near_miss_freqs_to_log[algo_name] = pm['smoothed_near_miss_frequency'][episode_idx]
                        safety_margins_to_log[algo_name] = pm['smoothed_safety_margin'][episode_idx]
                        comfort_scores_to_log[algo_name] = pm['smoothed_comfort_score'][episode_idx]

                if rewards_to_log: # Only log if there's data for this episode
                    self.writer.add_scalars(f'Comparative/Smoothed_Episode_Reward', rewards_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Episode_Length', lengths_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Smoothed_Success_Rate', success_rates_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Smoothed_Collision_Rate', collision_rates_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Smoothed_Near_Miss_Frequency', near_miss_freqs_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Smoothed_Safety_Margin', safety_margins_to_log, episode_idx)
                    self.writer.add_scalars(f'Comparative/Smoothed_Comfort_Score', comfort_scores_to_log, episode_idx)

        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        # Save results
        self.save_results(all_results)
        
        self.writer.close() # Close TensorBoard writer
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
                'Near-miss Freq': f"{metrics['near_miss_frequency']:.2%}", # Added
                'Mean Reward': f"{metrics['mean_reward']:.2f}",
                'Performance Consistency': f"{metrics['performance_consistency']:.2f}", # Added
                'Mean Comfort Score': f"{metrics['mean_comfort_score']:.2f}", # Added
                'Episode Length': f"{metrics['mean_episode_length']:.1f}",
                'Completion Time': f"{metrics['mean_completion_time']:.1f}" if metrics['mean_completion_time'] else "N/A",
                'Safety Time': f"{metrics['mean_safety_achievement_time']:.1f}" if metrics['mean_safety_achievement_time'] else "N/A",
                'Efficiency': f"{metrics['reward_per_step']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Find best performers
        print("\n🏆 BEST PERFORMERS:")
        
        # Best safety (lowest collision rate and near-miss frequency)
        best_safety = min(results.items(), key=lambda x: x[1]['collision_rate'] + x[1]['near_miss_frequency'])
        print(f"   🛡️  Safest Model: {best_safety[0]} ({best_safety[1]['collision_rate']:.2%} collision rate, {best_safety[1]['near_miss_frequency']:.2%} near-miss frequency)")
        
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
        print("   • For safety-critical applications, use the model with lowest collision rate and near-miss frequency")
        print("   • For general performance, consider the model with highest success rate")
        print("   • For efficiency, use the model with best reward-per-step ratio")
        
    def save_results(self, results):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = os.path.join("projectResults", "evaluation_results") # Changed results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary CSV
        summary_data = []
        for algo, metrics in results.items():
            summary_data.append({
                'Algorithm': algo,
                'Success_Rate': metrics['success_rate'],
                'Collision_Rate': metrics['collision_rate'], 
                'Near_Miss_Frequency': metrics['near_miss_frequency'], # Added
                'Mean_Reward': metrics['mean_reward'],
                'Std_Reward': metrics['std_reward'],
                'Performance_Consistency': metrics['performance_consistency'], # Added
                'Mean_Comfort_Score': metrics['mean_comfort_score'], # Added
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
    print("Check projectResults/evaluation_results/ for detailed output files.") # Updated path


if __name__ == "__main__":
    main()
