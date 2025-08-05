#!/usr/bin/env python3
"""
Quick training test script for debugging and validation
Uses minimal timesteps for fast iteration during development
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import make_env
import time
import traceback

def quick_train_test(algorithm='PPO', timesteps=1000, use_custom_env=False):
    """
    Quick training test with minimal timesteps
    Perfect for debugging and validation
    """
    print(f"🚀 Quick Training Test: {algorithm}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Custom env: {use_custom_env}")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # Create environment
        print("📦 Creating environment...")
        env = make_env("roundabout-v0", custom=use_custom_env)
        env.reset(seed=42)  # Fixed seed for reproducibility
        
        # Initialize model
        print(f"🤖 Initializing {algorithm} model...")
        algorithms = {
            'PPO': PPO,
            'DQN': DQN, 
            'A2C': A2C
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        model = algorithms[algorithm](
            'MlpPolicy',
            env,
            verbose=1,
            seed=42
        )
        
        # Quick training
        print(f"🏃 Training for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps)
        
        # Quick evaluation
        print("📊 Evaluating performance...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=3, deterministic=True
        )
        
        # Test saving
        print("💾 Testing model saving...")
        save_path = f"scratch/quick_test_{algorithm.lower()}"
        model.save(save_path)
        print(f"   Saved to: {save_path}")
        
        # Test loading
        print("📥 Testing model loading...")
        loaded_model = algorithms[algorithm].load(save_path, env=env)
        print("   ✅ Successfully loaded model")
        
        # Test prediction
        print("🎯 Testing predictions...")
        obs, info = env.reset()
        for i in range(5):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward:.3f}")
            if done or truncated:
                break
        
        env.close()
        
        training_time = time.time() - start_time
        
        print("\n✅ QUICK TEST RESULTS:")
        print(f"   Algorithm: {algorithm}")
        print(f"   Training time: {training_time:.1f} seconds")
        print(f"   Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   Status: SUCCESS")
        
        return True, {
            'algorithm': algorithm,
            'timesteps': timesteps,
            'training_time': training_time,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED:")
        print(f"   Error: {e}")
        print(f"   Algorithm: {algorithm}")
        traceback.print_exc()
        return False, None

def test_all_algorithms(timesteps=1000):
    """Test all available algorithms quickly"""
    print("🧪 Testing All Algorithms")
    print("=" * 50)
    
    algorithms = ['PPO', 'DQN', 'A2C']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*20} {algo} {'='*20}")
        success, result = quick_train_test(algo, timesteps)
        results[algo] = {'success': success, 'data': result}
        
        if not success:
            print(f"⚠️  {algo} failed, but continuing with other algorithms...")
    
    # Summary
    print("\n" + "=" * 50)
    print("ALGORITHM TEST SUMMARY")
    print("=" * 50)
    
    successful_algos = []
    
    for algo, result in results.items():
        if result['success']:
            data = result['data']
            print(f"✅ {algo:>5}: {data['mean_reward']:6.3f} ± {data['std_reward']:5.3f} "
                  f"({data['training_time']:.1f}s)")
            successful_algos.append(algo)
        else:
            print(f"❌ {algo:>5}: FAILED")
    
    print(f"\n📊 {len(successful_algos)}/{len(algorithms)} algorithms working")
    
    if successful_algos:
        print("🎉 Your setup is working! Ready for full training.")
    else:
        print("⚠️  No algorithms worked. Check your environment setup.")
    
    return results

def stress_test(algorithm='PPO', rounds=3):
    """Run multiple quick training rounds to test stability"""
    print(f"🏋️ Stress Testing {algorithm}")
    print(f"Running {rounds} training rounds...")
    print("-" * 40)
    
    results = []
    
    for round_num in range(rounds):
        print(f"\n🔄 Round {round_num + 1}/{rounds}")
        success, result = quick_train_test(algorithm, timesteps=500)
        results.append({'round': round_num + 1, 'success': success, 'data': result})
        
        if not success:
            print(f"⚠️  Round {round_num + 1} failed")
    
    # Analysis
    successful_rounds = [r for r in results if r['success']]
    success_rate = len(successful_rounds) / rounds * 100
    
    print(f"\n📊 STRESS TEST RESULTS:")
    print(f"   Success rate: {success_rate:.1f}% ({len(successful_rounds)}/{rounds})")
    
    if successful_rounds:
        rewards = [r['data']['mean_reward'] for r in successful_rounds]
        times = [r['data']['training_time'] for r in successful_rounds]
        
        print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
        print(f"   Average time: {sum(times)/len(times):.1f}s")
        print("   ✅ Algorithm appears stable")
    else:
        print("   ❌ Algorithm appears unstable")
    
    return results

def main():
    """Main testing interface"""
    print("⚡ Quick Training Test Utility")
    print("=" * 50)
    print("Choose your test:")
    print("1. Quick single algorithm test")
    print("2. Test all algorithms")
    print("3. Stress test an algorithm")
    print("4. Custom test")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            algo = input("Algorithm (PPO/DQN/A2C) [PPO]: ").strip() or 'PPO'
            timesteps = int(input("Timesteps [1000]: ") or "1000")
            quick_train_test(algo.upper(), timesteps)
            
        elif choice == '2':
            timesteps = int(input("Timesteps per algorithm [1000]: ") or "1000")
            test_all_algorithms(timesteps)
            
        elif choice == '3':
            algo = input("Algorithm to stress test [PPO]: ").strip() or 'PPO'
            rounds = int(input("Number of rounds [3]: ") or "3")
            stress_test(algo.upper(), rounds)
            
        elif choice == '4':
            print("Running custom test with PPO, 2000 timesteps...")
            quick_train_test('PPO', 2000, use_custom_env=True)
            
        else:
            print("Invalid choice, running default test...")
            quick_train_test('PPO', 1000)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    main() 