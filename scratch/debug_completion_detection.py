#!/usr/bin/env python3
"""
Debugger to identify and test roundabout course completion detection
Helps ensure proper termination and maximum reward on completion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.environment import make_env, register_custom_env
import numpy as np
import time

def debug_completion_conditions():
    """Debug what triggers course completion in the roundabout environment"""
    print("🔍 ROUNDABOUT COMPLETION DETECTION DEBUG")
    print("=" * 60)
    
    # Test both environments
    environments = [
        ("Standard", lambda: gym.make("roundabout-v0")),
        ("Custom", lambda: make_env("roundabout-v0", custom=True))
    ]
    
    for env_name, env_factory in environments:
        print(f"\n🧪 Testing {env_name} Environment:")
        print("-" * 40)
        
        env = env_factory()
        
        # Run multiple episodes to see completion patterns
        completion_data = []
        for episode in range(5):
            obs, info = env.reset(seed=42 + episode)
            
            print(f"\n📍 Episode {episode + 1}:")
            print(f"   Initial info: {info}")
            
            step_count = 0
            episode_reward = 0
            max_steps = 500  # Prevent infinite loops
            
            while step_count < max_steps:
                # Try different action strategies
                if step_count < 50:
                    action = 3  # FASTER - try to move through roundabout
                elif step_count < 100:
                    action = 0  # LANE_LEFT - try lane changes
                elif step_count < 150:
                    action = 2  # LANE_RIGHT
                else:
                    action = 3  # FASTER again
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Log important state changes
                if step_count % 50 == 0:
                    vehicle_pos = getattr(env.unwrapped.vehicle, 'position', 'Unknown') if hasattr(env.unwrapped, 'vehicle') else 'Unknown'
                    vehicle_speed = getattr(env.unwrapped.vehicle, 'speed', 'Unknown') if hasattr(env.unwrapped, 'vehicle') else 'Unknown'
                    print(f"   Step {step_count}: pos={vehicle_pos}, speed={vehicle_speed:.2f}, reward={reward:.3f}")
                
                # Check for completion or collision
                if done or truncated:
                    outcome = "Unknown"
                    if info.get('crashed', False):
                        outcome = "Collision"
                    elif info.get('arrived', False):
                        outcome = "Completed"
                    elif truncated:
                        outcome = "Truncated/Timeout"
                    
                    completion_data.append({
                        'episode': episode + 1,
                        'steps': step_count,
                        'reward': episode_reward,
                        'outcome': outcome,
                        'done': done,
                        'truncated': truncated,
                        'final_info': info
                    })
                    
                    print(f"   🏁 Episode ended: {outcome}")
                    print(f"      Steps: {step_count}, Total Reward: {episode_reward:.2f}")
                    print(f"      Final info: {info}")
                    break
            
            if step_count >= max_steps:
                print(f"   ⏰ Episode reached max steps ({max_steps})")
                completion_data.append({
                    'episode': episode + 1,
                    'steps': step_count,
                    'reward': episode_reward,
                    'outcome': 'Max Steps',
                    'done': False,
                    'truncated': False,
                    'final_info': info
                })
        
        env.close()
        
        # Analyze completion patterns
        print(f"\n📊 {env_name} Environment Analysis:")
        successful_completions = [d for d in completion_data if d['outcome'] == 'Completed']
        collisions = [d for d in completion_data if d['outcome'] == 'Collision']
        
        print(f"   Successful completions: {len(successful_completions)}/5")
        print(f"   Collisions: {len(collisions)}/5")
        print(f"   Average steps to completion: {np.mean([d['steps'] for d in successful_completions]) if successful_completions else 'N/A'}")
        print(f"   Average completion reward: {np.mean([d['reward'] for d in successful_completions]) if successful_completions else 'N/A'}")

def test_manual_roundabout_navigation():
    """Test manual navigation to understand completion triggers"""
    print(f"\n🎮 MANUAL ROUNDABOUT NAVIGATION TEST")
    print("=" * 60)
    
    register_custom_env()
    env = gym.make('custom-roundabout-v0', render_mode="human")
    
    print("🚗 Manual navigation test:")
    print("   This will show the roundabout visually")
    print("   Watch for when 'arrived' becomes True in info")
    
    # Test with a strategic action sequence
    action_sequence = [
        (3, 20, "Accelerate to enter roundabout"),
        (1, 5, "Brief pause to observe"),
        (0, 10, "Change to left lane"),
        (3, 30, "Continue around roundabout"),
        (2, 10, "Change back to right lane"),
        (3, 20, "Exit roundabout"),
        (3, 50, "Continue straight to complete"),
    ]
    
    obs, info = env.reset(seed=42)
    total_reward = 0
    total_steps = 0
    
    print(f"\n🎯 Executing strategic action sequence:")
    
    for action, duration, description in action_sequence:
        print(f"\n   {description} (action {action} for {duration} steps)")
        
        for step in range(duration):
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            
            # Check vehicle position and status
            if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
                pos = env.unwrapped.vehicle.position
                speed = env.unwrapped.vehicle.speed
                
                # Log every 10 steps
                if step % 10 == 0:
                    print(f"      Step {total_steps}: pos=({pos[0]:.1f}, {pos[1]:.1f}), speed={speed:.1f}, reward={reward:.3f}")
                
                # Check for important events
                if info.get('crashed', False):
                    print(f"      💥 COLLISION detected at step {total_steps}!")
                    break
                elif info.get('arrived', False):
                    print(f"      🎉 COMPLETION detected at step {total_steps}!")
                    print(f"         Total reward: {total_reward:.2f}")
                    break
            
            if done or truncated:
                print(f"      🏁 Episode terminated at step {total_steps}")
                print(f"         Done: {done}, Truncated: {truncated}")
                print(f"         Info: {info}")
                break
            
            time.sleep(0.05)  # Slow down for observation
        
        if done or truncated:
            break
    
    env.close()
    print(f"\n📊 Manual test completed:")
    print(f"   Total steps: {total_steps}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final info: {info}")

def analyze_highway_env_source():
    """Analyze the highway-env source to understand completion logic"""
    print(f"\n🔬 HIGHWAY-ENV SOURCE ANALYSIS")
    print("=" * 60)
    
    # Create environment to inspect its properties
    env = gym.make("roundabout-v0")
    
    print("🔍 Environment Properties:")
    print(f"   Environment class: {type(env.unwrapped).__name__}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Check environment config
    if hasattr(env.unwrapped, 'config'):
        config = env.unwrapped.config
        print(f"\n⚙️ Environment Configuration:")
        relevant_keys = ['duration', 'arrived_reward', 'collision_reward', 'simulation_frequency']
        for key in relevant_keys:
            if key in config:
                print(f"   {key}: {config[key]}")
    
    # Check for completion-related methods
    env_methods = [method for method in dir(env.unwrapped) if not method.startswith('_')]
    completion_methods = [method for method in env_methods if any(keyword in method.lower() for keyword in ['arrive', 'complete', 'goal', 'target'])]
    
    print(f"\n🎯 Potential completion-related methods:")
    for method in completion_methods:
        print(f"   {method}")
    
    # Test a quick episode to see internal state
    obs, info = env.reset()
    print(f"\n🚗 Initial state:")
    print(f"   Info: {info}")
    
    # Take a few steps to see how info changes
    for i in range(10):
        obs, reward, done, truncated, info = env.step(3)  # FASTER action
        if i % 3 == 0:
            print(f"   Step {i+1}: reward={reward:.3f}, done={done}, info={info}")
        if done or truncated:
            break
    
    env.close()

def test_completion_rewards():
    """Test different scenarios to see reward patterns"""
    print(f"\n💰 COMPLETION REWARD TESTING")
    print("=" * 60)
    
    register_custom_env()
    env = gym.make('custom-roundabout-v0')
    
    scenarios = [
        ("Quick completion attempt", [3] * 50),  # Fast forward
        ("Careful navigation", [3, 3, 0, 3, 3, 2, 3] * 10),  # Mixed actions
        ("Lane changing focus", [0, 2, 0, 2, 3] * 15),  # Focus on lanes
    ]
    
    for scenario_name, actions in scenarios:
        print(f"\n🎮 Testing: {scenario_name}")
        obs, info = env.reset(seed=42)
        total_reward = 0
        completion_reward = 0
        
        for step, action in enumerate(actions):
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if info.get('arrived', False):
                completion_reward = reward
                print(f"   🎉 COMPLETED at step {step+1}!")
                print(f"      Completion reward: {completion_reward:.2f}")
                print(f"      Total reward: {total_reward:.2f}")
                break
            elif info.get('crashed', False):
                print(f"   💥 CRASHED at step {step+1}")
                print(f"      Total reward: {total_reward:.2f}")
                break
        
        if not (done or truncated):
            print(f"   ⏰ Scenario completed without termination")
            print(f"      Total reward: {total_reward:.2f}")
    
    env.close()

def main():
    """Run all completion detection tests"""
    print("🚗 ROUNDABOUT COMPLETION DEBUGGER")
    print("This will help identify and fix course completion detection")
    print("=" * 60)
    
    try:
        analyze_highway_env_source()
        debug_completion_conditions()
        test_completion_rewards()
        
        print("\n" + "=" * 60)
        print("🔧 RECOMMENDATIONS FOR COMPLETION DETECTION:")
        print("=" * 60)
        print("1. Check if 'arrived' flag is properly set in highway-env")
        print("2. Ensure termination occurs immediately on completion")
        print("3. Verify completion reward is the highest possible")
        print("4. Add custom completion detection if needed")
        
        response = input("\n🎮 Run manual navigation test? (y/n): ")
        if response.lower() == 'y':
            test_manual_roundabout_navigation()
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 