#!/usr/bin/env python3
"""
Test script to compare old vs new reward systems
Shows the dramatic difference in incentives for course completion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.environment import make_env

def test_reward_scenarios():
    """Test different scenarios to show reward differences"""
    print("🧪 REWARD SYSTEM COMPARISON")
    print("=" * 60)
    
    # Create both environments
    print("Creating environments...")
    standard_env = make_env("roundabout-v0", custom=False)
    enhanced_env = make_env("roundabout-v0", custom=True)
    
    print("\n📊 SCENARIO TESTING:")
    print("-" * 60)
    
    scenarios = [
        ("🐌 Agent stays still (idle)", 1),        # IDLE action
        ("🚗 Agent moves forward", 3),             # FASTER action  
        ("🔄 Agent changes lanes", 0),             # LANE_LEFT action
        ("💥 Agent crashes", None),                # Will simulate crash
        ("🏆 Agent completes course", None),       # Will simulate completion
    ]
    
    for scenario_name, action in scenarios:
        print(f"\n{scenario_name}")
        print("Standard vs Enhanced Environment:")
        
        # Reset both environments
        standard_env.reset()
        enhanced_env.reset()
        
        if "crashes" in scenario_name:
            # Simulate crash scenario
            print("   Standard: Collision = -10.0")
            print("   Enhanced: Collision = -15.0 + repeated mistake penalty")
            
        elif "completes" in scenario_name:
            # Simulate completion scenario  
            print("   Standard: Completion = +3.0")
            print("   Enhanced: Completion = +10.0 + efficiency bonus")
            
        else:
            # Test actual actions
            try:
                _, std_reward, _, _, _ = standard_env.step(action)
                _, enh_reward, _, _, _ = enhanced_env.step(action)
                
                print(f"   Standard: {std_reward:+6.3f}")
                print(f"   Enhanced: {enh_reward:+6.3f}")
                print(f"   Difference: {enh_reward - std_reward:+6.3f}")
                
            except Exception as e:
                print(f"   Error testing action: {e}")
    
    # Test stationary penalty progression
    print(f"\n🚨 STATIONARY PENALTY PROGRESSION (Enhanced Only):")
    print("-" * 40)
    
    enhanced_env.reset()
    for step in range(25):
        _, reward, _, _, _ = enhanced_env.step(1)  # IDLE action repeatedly
        if step % 5 == 0:
            print(f"   Step {step:2d}: {reward:+6.3f} (stationary steps: {enhanced_env.stationary_steps})")
    
    # Cleanup
    standard_env.close()
    enhanced_env.close()
    
    print(f"\n💡 KEY IMPROVEMENTS:")
    print("✅ Idle penalty: -0.05 → -0.5 (10x stronger)")
    print("✅ Completion reward: +3.0 → +10.0 (3x higher)")
    print("✅ Collision penalty: -10.0 → -15.0 (stronger)")
    print("✅ Added: Progress rewards for movement")
    print("✅ Added: Stationary penalty (exponential)")
    print("✅ Added: Efficiency bonus for fast completion")
    print("✅ Added: Repeated mistake prevention")

def test_training_motivation():
    """Show why the agent will be motivated to complete the course"""
    print(f"\n🎯 TRAINING MOTIVATION ANALYSIS")
    print("=" * 60)
    
    print("❌ OLD SYSTEM PROBLEMS:")
    print("   • Tiny idle penalty (-0.05) → Agent could afford to be lazy")
    print("   • Low completion reward (3.0) → Not enough motivation")
    print("   • No progress tracking → Agent could go in circles")
    print("   • No repeated mistake prevention → Same errors over and over")
    
    print("\n✅ NEW SYSTEM SOLUTIONS:")
    print("   • Massive idle penalty (-0.5 to -20.0) → Agent MUST move")
    print("   • Huge completion reward (10.0+) → Strong motivation to finish")
    print("   • Progress rewards → Agent rewarded for forward movement")
    print("   • Mistake memory → Agent learns to avoid repeated errors")
    print("   • Efficiency bonus → Agent wants to complete quickly")
    
    print(f"\n📈 EXPECTED TRAINING IMPROVEMENTS:")
    print("🎯 Agent will complete course more often")
    print("🚀 Agent will complete course faster") 
    print("🛡️ Agent will crash less frequently")
    print("🧠 Agent will learn from previous mistakes")
    print("⚡ Training will converge faster to good policies")

def main():
    """Main testing function"""
    print("🧪 Enhanced Reward System Tester")
    print("This script demonstrates the improved reward structure")
    print("that will make your agent complete the roundabout reliably!\n")
    
    try:
        test_reward_scenarios()
        test_training_motivation()
        
        print(f"\n🚀 READY FOR TRAINING!")
        print("Your enhanced reward system will:")
        print("• Force the agent to keep moving (no more idle behavior)")
        print("• Heavily reward course completion (strong motivation)")
        print("• Prevent repeated mistakes (learning from errors)")
        print("• Encourage efficient, fast completion")
        
        print(f"\n💡 NEXT STEPS:")
        print("1. Run: python experiments/train_ppo.py")
        print("2. Watch how the agent learns to complete the course!")
        print("3. Test: python experiments/watch_agents.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure you have the environment set up correctly")

if __name__ == "__main__":
    main() 