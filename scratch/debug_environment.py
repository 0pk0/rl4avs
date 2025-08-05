#!/usr/bin/env python3
"""
Debug script for testing environment setup and configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import highway_env
import traceback
from src.environment import make_env, CustomRoundaboutEnv
import numpy as np

def test_basic_environment():
    """Test basic highway-env installation and roundabout environment"""
    print("🔧 Testing Basic Environment Setup")
    print("=" * 50)
    
    try:
        # Test standard roundabout environment
        print("Testing standard roundabout-v0...")
        env = gym.make("roundabout-v0")
        obs, info = env.reset()
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Initial observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
            if done or truncated:
                obs, info = env.reset()
                break
        
        env.close()
        print("✅ Basic environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        traceback.print_exc()
        return False

def test_custom_environment():
    """Test custom environment implementation"""
    print("🔧 Testing Custom Environment")
    print("=" * 50)
    
    try:
        # Test custom environment
        print("Testing custom roundabout environment...")
        env = make_env("roundabout-v0", custom=True)
        obs, info = env.reset()
        print(f"✅ Custom environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test custom reward modifications
        print("Testing custom reward mechanics...")
        rewards = []
        actions_tested = [0, 1, 2, 3, 4]  # All possible actions
        
        for action in actions_tested:
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            action_name = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
            print(f"   Action {action} ({action_name[action]}): reward={reward:.3f}")
            
            if done or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✅ Custom environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Custom environment test failed: {e}")
        traceback.print_exc()
        return False

def test_rendering():
    """Test environment rendering capabilities"""
    print("🔧 Testing Rendering Capabilities")
    print("=" * 50)
    
    try:
        print("Testing human rendering mode...")
        env = gym.make("roundabout-v0", render_mode="human")
        obs, info = env.reset()
        
        print("✅ Rendering environment created")
        print("   Running 10 steps with visualization...")
        print("   (Close the pygame window when done viewing)")
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            if done or truncated:
                obs, info = env.reset()
                
        env.close()
        print("✅ Rendering test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Rendering test failed: {e}")
        print("   This might be normal if running in headless mode")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("🔧 Testing Dependencies")
    print("=" * 50)
    
    dependencies = {
        'gymnasium': 'gymnasium',
        'highway_env': 'highway-env', 
        'stable_baselines3': 'stable-baselines3',
        'torch': 'torch',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'pygame': 'pygame'
    }
    
    failed_imports = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("Install with: pip install " + " ".join(failed_imports))
        return False
    else:
        print("\n✅ All dependencies installed correctly!")
        return True

def diagnose_model_saving():
    """Test model saving and loading capabilities"""
    print("🔧 Testing Model Saving/Loading")
    print("=" * 50)
    
    try:
        from stable_baselines3 import PPO
        
        # Test directory creation
        test_dirs = [
            "scratch/test_models",
            "results/models", 
            "experiments/results/models"
        ]
        
        for test_dir in test_dirs:
            os.makedirs(test_dir, exist_ok=True)
            print(f"✅ Created directory: {test_dir}")
        
        # Test model creation and saving
        print("Creating test PPO model...")
        env = make_env("roundabout-v0", custom=False)
        model = PPO('MlpPolicy', env, verbose=0)
        
        # Test saving to different locations
        test_paths = [
            "scratch/test_models/test_ppo",
            "results/models/test_ppo", 
            "experiments/results/models/test_ppo"
        ]
        
        for test_path in test_paths:
            try:
                model.save(test_path)
                print(f"✅ Saved model to: {test_path}")
                
                # Test loading
                loaded_model = PPO.load(test_path, env=env)
                print(f"✅ Loaded model from: {test_path}")
                
            except Exception as e:
                print(f"❌ Failed to save/load model at {test_path}: {e}")
        
        env.close()
        print("✅ Model saving/loading test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model saving test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 RL4AVS Environment Diagnostics")
    print("=" * 60)
    print("This script will test your environment setup and identify issues")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Basic Environment", test_basic_environment),
        ("Custom Environment", test_custom_environment),
        ("Model Saving", diagnose_model_saving),
        ("Rendering", test_rendering),  # Last since it opens windows
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⚠️  Test {test_name} interrupted by user")
            results[test_name] = False
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for training.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("💡 Common fixes:")
        print("   - Install missing dependencies with pip")
        print("   - Check file permissions for saving models")
        print("   - Ensure display is available for rendering")

if __name__ == "__main__":
    main() 