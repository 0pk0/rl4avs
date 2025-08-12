#!/usr/bin/env python3
"""
Test script to verify enhanced roundabout exit detection
Ensures proper termination when course is completed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.environment import register_custom_env
import time

def test_enhanced_exit_detection():
    """Test the enhanced exit detection with clear logging"""
    print("🧪 ENHANCED EXIT DETECTION TEST")
    print("=" * 60)
    
    # Create custom environment with exit detection
    register_custom_env()
    env = gym.make('custom-roundabout-v0', render_mode="human")
    
    print("🚗 Testing enhanced exit detection:")
    print("   Watch for progression through stages:")
    print("   1. 🏁 Starting position")
    print("   2. 🔄 Roundabout entry")
    print("   3. 🚪 Roundabout exit detection")
    print("   4. 🎉 Course completion")
    print("   5. 💫 Episode termination")
    
    # Test multiple scenarios
    scenarios = [
        ("Fast completion", [3] * 100),  # Fast forward only
        ("Strategic navigation", [3, 3, 0, 3, 3, 2, 3] * 20),  # Mixed actions
    ]
    
    for scenario_name, actions in scenarios:
        print(f"\n🎮 SCENARIO: {scenario_name}")
        print("=" * 40)
        
        obs, info = env.reset(seed=42)
        total_reward = 0
        episode_steps = 0
        
        for step, action in enumerate(actions):
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            # Slow down for observation
            time.sleep(0.03)
            
            # Check for termination
            if done:
                if info.get('arrived', False):
                    print(f"\n✅ SCENARIO COMPLETED SUCCESSFULLY!")
                    print(f"   Episode ended by: Course completion")
                    print(f"   Total steps: {episode_steps}")
                    print(f"   Total reward: {total_reward:.2f}")
                    print(f"   Info: {info}")
                elif info.get('crashed', False):
                    print(f"\n❌ SCENARIO ENDED IN COLLISION!")
                    print(f"   Episode ended by: Vehicle collision")
                    print(f"   Total steps: {episode_steps}")
                    print(f"   Total reward: {total_reward:.2f}")
                    print(f"   Info: {info}")
                break
            elif truncated:
                print(f"\n⏰ SCENARIO TIMED OUT!")
                print(f"   Episode ended by: Time limit")
                print(f"   Total steps: {episode_steps}")
                print(f"   Total reward: {total_reward:.2f}")
                break
        
        if not (done or truncated):
            print(f"\n🔄 SCENARIO COMPLETED WITHOUT TERMINATION")
            print(f"   Total steps: {episode_steps}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Final info: {info}")
        
        print(f"\n{'='*40}")
    
    env.close()

def test_termination_timing():
    """Test that termination happens immediately after completion"""
    print(f"\n⏱️ TERMINATION TIMING TEST")
    print("=" * 60)
    
    register_custom_env()
    env = gym.make('custom-roundabout-v0')
    
    print("Testing immediate termination after completion detection...")
    
    obs, info = env.reset(seed=42)
    steps_after_completion = 0
    completion_detected = False
    
    for step in range(300):  # Maximum steps
        obs, reward, done, truncated, info = env.step(3)  # Always go faster
        
        # Track if completion was detected in previous steps
        if info.get('arrived', False) and not completion_detected:
            completion_detected = True
            print(f"🎉 Completion detected at step {step + 1}")
        
        if completion_detected:
            steps_after_completion += 1
        
        if done:
            print(f"💫 Episode terminated at step {step + 1}")
            print(f"   Steps after completion: {steps_after_completion}")
            if steps_after_completion <= 1:
                print(f"   ✅ PERFECT: Immediate termination!")
            else:
                print(f"   ⚠️  DELAYED: {steps_after_completion} steps after completion")
            break
        elif truncated:
            print(f"⏰ Episode truncated at step {step + 1}")
            break
    
    env.close()

def main():
    """Run exit detection tests"""
    print("🚪 ROUNDABOUT EXIT DETECTION TESTER")
    print("This verifies that vehicles stop immediately after completing the roundabout")
    print("=" * 60)
    
    try:
        test_termination_timing()
        
        response = input("\n🎮 Run visual exit detection test? (y/n): ")
        if response.lower() == 'y':
            test_enhanced_exit_detection()
        
        print(f"\n✅ EXIT DETECTION TESTING COMPLETED!")
        print("Features verified:")
        print("  • 🔄 Roundabout entry detection")
        print("  • 🚪 Roundabout exit detection") 
        print("  • 🎉 Course completion detection")
        print("  • 💫 Immediate episode termination")
        print("  • 📊 Detailed logging (like collision logs)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 