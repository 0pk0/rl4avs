#!/usr/bin/env python3
"""
Script to test loading and evaluating saved models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import glob
from pathlib import Path
import numpy as np

def find_saved_models():
    """Find all saved models in the project"""
    print("🔍 Searching for saved models...")
    
    model_dirs = [
        "results/models/",
        "experiments/results/models/",
        "scratch/test_models/"
    ]
    
    found_models = {}
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Look for .zip files (stable-baselines3 format)
            zip_files = glob.glob(os.path.join(model_dir, "*.zip"))
            for zip_file in zip_files:
                model_name = os.path.basename(zip_file).replace('.zip', '')
                found_models[model_name] = zip_file
                print(f"  Found: {model_name} at {zip_file}")
    
    print(f"📊 Total models found: {len(found_models)}")
    return found_models

def load_and_test_model(model_path, model_name):
    """Load a model and test it briefly"""
    print(f"\n🧪 Testing model: {model_name}")
    print("-" * 40)
    
    try:
        # Determine algorithm type from name
        if 'PPO' in model_name.upper():
            model = PPO.load(model_path)
            algo_type = "PPO"
        elif 'DQN' in model_name.upper():
            model = DQN.load(model_path)
            algo_type = "DQN"
        elif 'A2C' in model_name.upper():
            model = A2C.load(model_path)
            algo_type = "A2C"
        else:
            # Try PPO as default
            model = PPO.load(model_path)
            algo_type = "PPO (assumed)"
        
        print(f"✅ Successfully loaded {algo_type} model")
        
        # Test prediction capability
        env = gym.make("roundabout-v0")
        obs, info = env.reset()
        
        # Test predictions
        actions = []
        rewards = []
        
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            
            if done or truncated:
                obs, info = env.reset()
        
        env.close()
        
        print(f"   Actions sampled: {actions}")
        print(f"   Average reward: {np.mean(rewards):.3f}")
        print(f"   Total reward: {np.sum(rewards):.3f}")
        
        return True, algo_type, np.mean(rewards)
        
    except Exception as e:
        print(f"❌ Failed to load/test model: {e}")
        return False, None, None

def evaluate_model_performance(model_path, model_name, n_episodes=5):
    """Evaluate model performance over multiple episodes"""
    print(f"\n📈 Evaluating {model_name} over {n_episodes} episodes...")
    
    try:
        # Load model
        if 'PPO' in model_name.upper():
            model = PPO.load(model_path)
        elif 'DQN' in model_name.upper():
            model = DQN.load(model_path)
        elif 'A2C' in model_name.upper():
            model = A2C.load(model_path)
        else:
            model = PPO.load(model_path)
        
        env = gym.make("roundabout-v0")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        collision_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Count outcomes
            if info.get('crashed', False):
                collision_count += 1
            elif info.get('arrived', False):
                success_count += 1
        
        env.close()
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = success_count / n_episodes * 100
        collision_rate = collision_count / n_episodes * 100
        
        print(f"   Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   Mean episode length: {mean_length:.1f} steps")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Collision rate: {collision_rate:.1f}%")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'success_rate': success_rate,
            'collision_rate': collision_rate
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def compare_models(model_results):
    """Compare performance across models"""
    if len(model_results) < 2:
        print("Need at least 2 models for comparison")
        return
    
    print("\n📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Reward':<12} {'Success%':<10} {'Collisions%':<12}")
    print("-" * 60)
    
    for model_name, results in model_results.items():
        if results:
            print(f"{model_name:<25} {results['mean_reward']:7.3f}     "
                  f"{results['success_rate']:6.1f}     {results['collision_rate']:8.1f}")
    
    # Find best model
    best_reward = max(model_results.items(), 
                     key=lambda x: x[1]['mean_reward'] if x[1] else -float('inf'))
    best_success = max(model_results.items(), 
                      key=lambda x: x[1]['success_rate'] if x[1] else -1)
    
    print("\n🏆 BEST PERFORMERS:")
    print(f"   Highest reward: {best_reward[0]} ({best_reward[1]['mean_reward']:.3f})")
    print(f"   Highest success rate: {best_success[0]} ({best_success[1]['success_rate']:.1f}%)")

def main():
    """Main testing function"""
    print("🧪 Model Loading and Testing Utility")
    print("=" * 50)
    
    # Find all models
    models = find_saved_models()
    
    if not models:
        print("❌ No saved models found!")
        print("   Train some models first using train_agents.py")
        return
    
    print(f"\n📋 Testing {len(models)} found models...")
    
    # Basic load tests
    working_models = {}
    model_results = {}
    
    for model_name, model_path in models.items():
        success, algo_type, avg_reward = load_and_test_model(model_path, model_name)
        if success:
            working_models[model_name] = (model_path, algo_type)
    
    print(f"\n✅ {len(working_models)}/{len(models)} models loaded successfully")
    
    if not working_models:
        print("❌ No models could be loaded successfully")
        return
    
    # Detailed evaluation
    print("\n🎯 Running detailed evaluation...")
    for model_name, (model_path, algo_type) in working_models.items():
        results = evaluate_model_performance(model_path, model_name, n_episodes=3)
        model_results[model_name] = results
    
    # Compare models
    compare_models(model_results)
    
    print("\n🎉 Model testing completed!")
    print("💡 To watch a model in action, use: python experiments/watch_agents.py")

if __name__ == "__main__":
    main() 