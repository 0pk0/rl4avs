#!/usr/bin/env python3
"""
Diagnostic script to debug reward issues during training
Helps identify if custom environment is actually being used
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.environment import make_env, register_custom_env
from stable_baselines3 import PPO
import numpy as np

def test_environment_rewards():
    """Test both environments to see actual reward differences"""
    print("🔍 ENVIRONMENT REWARD TESTING")
    print("=" * 60)
    
    # Test standard environment
    print("\n1️⃣ TESTING STANDARD ENVIRONMENT:")
    standard_env = gym.make("roundabout-v0")
    standard_env.reset(seed=42)
    
    standard_rewards = []
    for i in range(10):
        action = 1 if i < 5 else 3  # First 5 idle, then moving
        obs, reward, done, truncated, info = standard_env.step(action)
        standard_rewards.append(reward)
        action_name = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        print(f"   Step {i+1}: {action_name[action]:10} → Reward: {reward:+6.3f}")
        if done or truncated:
            obs, info = standard_env.reset(seed=42)
    
    standard_env.close()
    
    # Test custom environment  
    print("\n2️⃣ TESTING CUSTOM ENVIRONMENT:")
    register_custom_env()
    custom_env = gym.make('custom-roundabout-v0')
    custom_env.reset(seed=42)
    
    custom_rewards = []
    for i in range(10):
        action = 1 if i < 5 else 3  # First 5 idle, then moving
        obs, reward, done, truncated, info = custom_env.step(action)
        custom_rewards.append(reward)
        action_name = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        print(f"   Step {i+1}: {action_name[action]:10} → Reward: {reward:+6.3f}")
        if done or truncated:
            obs, info = custom_env.reset(seed=42)
    
    custom_env.close()
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print(f"Standard avg reward: {np.mean(standard_rewards):+6.3f}")
    print(f"Custom avg reward:   {np.mean(custom_rewards):+6.3f}")
    print(f"Difference:          {np.mean(custom_rewards) - np.mean(standard_rewards):+6.3f}")
    
    if abs(np.mean(custom_rewards) - np.mean(standard_rewards)) < 0.1:
        print("❌ WARNING: Rewards are too similar! Custom environment may not be working!")
    else:
        print("✅ Rewards are different - custom environment is working!")

def test_environment_creation():
    """Test how environments are created with different settings"""
    print(f"\n🔧 ENVIRONMENT CREATION TESTING")
    print("=" * 60)
    
    # Test make_env function
    print("\n🧪 Testing make_env function:")
    try:
        env1 = make_env("roundabout-v0", custom=False)
        print("✅ Standard environment created successfully")
        env1.close()
    except Exception as e:
        print(f"❌ Standard environment failed: {e}")
    
    try:
        env2 = make_env("roundabout-v0", custom=True)
        print("✅ Custom environment created successfully")
        # Check if it has custom attributes
        if hasattr(env2.unwrapped, 'mistake_memory'):
            print("   ✅ Custom tracking variables detected")
        else:
            print("   ❌ Custom tracking variables NOT found")
        env2.close()
    except Exception as e:
        print(f"❌ Custom environment failed: {e}")
    
    # Test direct gym.make calls
    print("\n🎯 Testing direct gym.make calls:")
    try:
        env3 = gym.make("roundabout-v0")
        print("✅ Direct standard environment created")
        env3.close()
    except Exception as e:
        print(f"❌ Direct standard environment failed: {e}")
    
    try:
        register_custom_env()
        env4 = gym.make('custom-roundabout-v0')
        print("✅ Direct custom environment created")
        if hasattr(env4.unwrapped, 'mistake_memory'):
            print("   ✅ Custom tracking variables detected")
        else:
            print("   ❌ Custom tracking variables NOT found")
        env4.close()
    except Exception as e:
        print(f"❌ Direct custom environment failed: {e}")

def simulate_training_setup():
    """Simulate the exact training setup to see what happens"""
    print(f"\n🚀 SIMULATING TRAINING SETUP")
    print("=" * 60)
    
    env_name = "roundabout-v0"
    use_custom_env = True
    show_training = True
    
    print(f"Settings: use_custom_env={use_custom_env}, show_training={show_training}")
    
    # Original buggy code path
    print("\n❌ ORIGINAL BUGGY CODE PATH:")
    train_env = make_env(env_name, custom=use_custom_env)
    print(f"   1. Created: {type(train_env.unwrapped).__name__}")
    
    if show_training:
        train_env.close()
        train_env = gym.make(env_name, render_mode="human")
        print(f"   2. Overwritten with: {type(train_env.unwrapped).__name__}")
    
    train_env.close()
    
    # Fixed code path
    print("\n✅ FIXED CODE PATH:")
    if show_training and use_custom_env:
        register_custom_env()
        train_env = gym.make('custom-roundabout-v0', render_mode="human")
        print(f"   Created: {type(train_env.unwrapped).__name__}")
    elif show_training:
        train_env = gym.make(env_name, render_mode="human")
        print(f"   Created: {type(train_env.unwrapped).__name__}")
    else:
        train_env = make_env(env_name, custom=use_custom_env)
        print(f"   Created: {type(train_env.unwrapped).__name__}")
    
    if hasattr(train_env.unwrapped, 'mistake_memory'):
        print("   ✅ Custom environment confirmed!")
    else:
        print("   ❌ Standard environment - bug not fixed!")
    
    train_env.close()

def quick_training_test():
    """Quick test to see what rewards we actually get during training"""
    print(f"\n⚡ QUICK TRAINING TEST")
    print("=" * 60)
    
    print("Testing 1000 steps with custom environment...")
    
    # Create custom environment
    register_custom_env()
    env = gym.make('custom-roundabout-v0')
    
    # Create simple PPO model
    model = PPO('MlpPolicy', env, verbose=0)
    
    # Test a few episodes to see rewards
    episode_rewards = []
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = truncated = False
        
        while not (done or truncated) and steps < 200:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        outcome = "Success" if info.get('arrived', False) else "Collision" if info.get('crashed', False) else "Timeout"
        print(f"   Episode {episode+1}: {episode_reward:+6.2f} ({steps} steps, {outcome})")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"\n📊 Average reward: {avg_reward:+6.2f}")
    
    if avg_reward < 1.0:
        print("❌ Rewards still too low! Agent likely not completing course.")
        print("   Possible issues:")
        print("   • Custom environment not working properly")  
        print("   • Penalties too harsh for learning")
        print("   • Agent needs more training time")
    else:
        print("✅ Rewards look reasonable for custom environment")

def main():
    """Run all diagnostic tests"""
    print("🔍 REWARD ISSUE DIAGNOSTIC")
    print("This script will help identify why avg reward is only 0.8")
    print("Expected custom reward: much higher due to +10.0 completion bonus\n")
    
    try:
        test_environment_creation()
        test_environment_rewards()
        simulate_training_setup()
        quick_training_test()
        
        print(f"\n💡 LIKELY CAUSE OF 0.8 REWARD:")
        print("The training was using STANDARD environment (weak rewards)")
        print("instead of CUSTOM environment (strong rewards)")
        print("\n🔧 SOLUTION:")
        print("Use the FIXED train_ppo.py code that properly maintains")
        print("custom environment even when show_training=True")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 