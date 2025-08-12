#!/usr/bin/env python3
"""
🔍 OBSERVATION DEBUGGER FOR AUTONOMOUS VEHICLE RL AGENT 🔍

This tool shows you EXACTLY what your agent "sees" in the observation space
right before it makes decisions. This is crucial for understanding why
collisions happen and what information the agent is missing or misinterpreting.

🎯 WHAT THIS REVEALS:
- Positions and velocities of all nearby vehicles (up to 15)
- Whether the agent can "see" vehicles that are about to collide with it
- How the agent interprets relative positions and speeds
- Missing information that might explain collision behavior

🚨 KEY INSIGHTS FOR COLLISION ANALYSIS:
- If nearby vehicles are present in observations but agent still crashes → Policy problem
- If nearby vehicles are NOT in observations before crash → Observation space limitation
- If vehicle data is confusing/unclear → Need better feature engineering

🔧 USAGE:
   python scratch/debug_observations.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from src.environment import register_custom_env, make_env
import time
import matplotlib.pyplot as plt
from collections import defaultdict

class ObservationDebugger:
    """Debug tool to analyze what the agent observes before making decisions"""
    
    def __init__(self, model_path=None, use_custom_env=True):
        self.use_custom_env = use_custom_env
        
        # Create environment
        if use_custom_env:
            register_custom_env()
            self.env = gym.make('custom-roundabout-v0', render_mode="human")
            print("🎁 Using CUSTOM environment for debugging")
        else:
            self.env = gym.make('roundabout-v0', render_mode="human")
            print("⚠️ Using STANDARD environment for debugging")
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path + ".zip"):
            try:
                self.model = PPO.load(model_path)
                print(f"✅ Loaded model from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🤖 Will use random actions for debugging")
        else:
            print("🤖 No model provided - using random actions for observation debugging")
        
        # Observation analysis
        self.collision_observations = []
        self.normal_observations = []
        self.action_history = []
        
    def analyze_observation(self, obs, step_count, action=None, collision_risk=False):
        """Analyze and print detailed observation breakdown"""
        
        print(f"\n{'🚨 COLLISION RISK' if collision_risk else '📊 NORMAL'} - Step {step_count}")
        print("=" * 60)
        
        # The observation is a flattened array representing up to 15 vehicles
        # Each vehicle has 7 features: [presence, x, y, vx, vy, cos_h, sin_h]
        obs_reshaped = obs.reshape(-1, 7)  # Reshape to [n_vehicles, 7_features]
        
        ego_info = obs_reshaped[0]  # First row is ego vehicle
        other_vehicles = obs_reshaped[1:]  # Rest are other vehicles
        
        print(f"🚗 EGO VEHICLE:")
        print(f"   Position: ({ego_info[1]:.2f}, {ego_info[2]:.2f})")
        print(f"   Velocity: ({ego_info[3]:.2f}, {ego_info[4]:.2f}) | Speed: {np.sqrt(ego_info[3]**2 + ego_info[4]**2):.2f}")
        print(f"   Heading: cos={ego_info[5]:.2f}, sin={ego_info[6]:.2f}")
        
        # Analyze other vehicles
        present_vehicles = []
        for i, vehicle in enumerate(other_vehicles):
            if vehicle[0] > 0.5:  # Vehicle is present
                distance = np.sqrt(vehicle[1]**2 + vehicle[2]**2)
                speed = np.sqrt(vehicle[3]**2 + vehicle[4]**2)
                present_vehicles.append({
                    'id': i+1,
                    'pos': (vehicle[1], vehicle[2]),
                    'vel': (vehicle[3], vehicle[4]),
                    'distance': distance,
                    'speed': speed,
                    'heading': (vehicle[5], vehicle[6])
                })
        
        # Sort by distance (closest first)
        present_vehicles.sort(key=lambda x: x['distance'])
        
        print(f"\n🚙 OTHER VEHICLES DETECTED: {len(present_vehicles)}/15")
        
        if len(present_vehicles) == 0:
            print("   ⚠️ NO OTHER VEHICLES DETECTED!")
            if collision_risk:
                print("   🚨 CRITICAL: Collision risk but no vehicles seen - BLIND SPOT ISSUE!")
        
        # Show closest vehicles with danger assessment
        for i, vehicle in enumerate(present_vehicles[:5]):  # Show top 5 closest
            danger_level = self._assess_danger(ego_info, vehicle)
            danger_emoji = "🚨" if danger_level > 0.7 else "⚠️" if danger_level > 0.4 else "✅"
            
            print(f"   {danger_emoji} Vehicle {vehicle['id']} (Danger: {danger_level:.2f})")
            print(f"      Distance: {vehicle['distance']:.2f}m")
            print(f"      Relative Pos: ({vehicle['pos'][0]:.2f}, {vehicle['pos'][1]:.2f})")
            print(f"      Relative Vel: ({vehicle['vel'][0]:.2f}, {vehicle['vel'][1]:.2f})")
            print(f"      Speed: {vehicle['speed']:.2f} m/s")
        
        if action is not None:
            action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
            print(f"\n🎮 AGENT ACTION: {action_names[action]} ({action})")
            
            # Assess if action is appropriate
            if len(present_vehicles) > 0:
                closest = present_vehicles[0]
                if closest['distance'] < 10 and action == 1:  # IDLE when close vehicle
                    print("   ⚠️ RISKY: Staying idle near close vehicle!")
                elif closest['distance'] < 5 and action == 3:  # FASTER when very close
                    print("   🚨 DANGEROUS: Accelerating toward close vehicle!")
        
        print("-" * 60)
        
        # Store for pattern analysis
        obs_data = {
            'observation': obs.copy(),
            'step': step_count,
            'n_vehicles': len(present_vehicles),
            'closest_distance': present_vehicles[0]['distance'] if present_vehicles else float('inf'),
            'action': action,
            'collision_risk': collision_risk
        }
        
        if collision_risk:
            self.collision_observations.append(obs_data)
        else:
            self.normal_observations.append(obs_data)
    
    def _assess_danger(self, ego_info, other_vehicle):
        """Calculate danger level (0-1) based on distance, relative velocity, heading"""
        
        # Distance factor (closer = more dangerous)
        distance = other_vehicle['distance']
        distance_factor = max(0, 1 - distance / 20)  # Dangerous if < 20m
        
        # Relative velocity factor (approaching = more dangerous)
        rel_vel_x, rel_vel_y = other_vehicle['vel']
        ego_vel_x, ego_vel_y = ego_info[3], ego_info[4]
        
        # Check if vehicles are moving toward each other
        approaching = (rel_vel_x * other_vehicle['pos'][0] + rel_vel_y * other_vehicle['pos'][1]) < 0
        vel_factor = 1.0 if approaching else 0.3
        
        # Combine factors
        danger = distance_factor * vel_factor
        return min(1.0, danger)
    
    def run_debugging_session(self, max_episodes=3, max_steps_per_episode=300):
        """Run debugging session with detailed observation analysis"""
        
        print("🔍 STARTING OBSERVATION DEBUGGING SESSION")
        print("=" * 60)
        print("🎯 GOAL: Understand what agent sees before making decisions")
        print("🚨 FOCUS: Identify why collisions happen despite available information")
        print("")
        
        for episode in range(max_episodes):
            print(f"\n🎬 EPISODE {episode + 1}")
            obs, info = self.env.reset()
            episode_reward = 0
            collision_detected = False
            
            for step in range(max_steps_per_episode):
                # Determine if we're in a risky situation
                # (This is a heuristic - you can improve this logic)
                obs_reshaped = obs.reshape(-1, 7)
                other_vehicles = obs_reshaped[1:]
                collision_risk = False
                for vehicle in other_vehicles:
                    if vehicle[0] > 0.5:  # Vehicle present
                        distance = np.sqrt(vehicle[1]**2 + vehicle[2]**2)
                        if distance < 8:  # Very close vehicle
                            collision_risk = True
                            break
                
                # Get action
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Analyze observation before step
                if collision_risk or step % 20 == 0:  # Show risky situations + periodic updates
                    self.analyze_observation(obs, step, action, collision_risk)
                
                # Take step
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Check for collision
                if info.get('crashed', False):
                    collision_detected = True
                    print(f"\n💥 COLLISION DETECTED at step {step}!")
                    print(f"   Final reward: {reward:.3f}")
                    print(f"   Episode reward: {episode_reward:.3f}")
                    
                    # Show final observation that led to collision
                    print(f"\n🔍 FINAL OBSERVATION BEFORE COLLISION:")
                    self.analyze_observation(obs, step, action, True)
                    break
                
                if done or truncated:
                    if info.get('arrived', False):
                        print(f"\n🎉 SUCCESSFUL COMPLETION at step {step}!")
                    else:
                        print(f"\n⏱️ Episode ended (timeout) at step {step}")
                    print(f"   Episode reward: {episode_reward:.3f}")
                    break
                
                time.sleep(0.05)  # Slow down for observation
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"   Collision: {'YES' if collision_detected else 'NO'}")
            print(f"   Total Steps: {step + 1}")
            print(f"   Total Reward: {episode_reward:.3f}")
            
            input("\n⏸️ Press Enter to continue to next episode...")
    
    def analyze_collision_patterns(self):
        """Analyze patterns in observations that lead to collisions"""
        
        if len(self.collision_observations) == 0:
            print("📊 No collision observations recorded yet")
            return
        
        print(f"\n📊 COLLISION PATTERN ANALYSIS")
        print("=" * 60)
        print(f"Total collision observations: {len(self.collision_observations)}")
        print(f"Total normal observations: {len(self.normal_observations)}")
        
        # Analyze vehicle detection rates
        collision_vehicle_counts = [obs['n_vehicles'] for obs in self.collision_observations]
        normal_vehicle_counts = [obs['n_vehicles'] for obs in self.normal_observations]
        
        print(f"\n🚙 VEHICLE DETECTION ANALYSIS:")
        print(f"   Avg vehicles seen before collision: {np.mean(collision_vehicle_counts):.1f}")
        print(f"   Avg vehicles seen normally: {np.mean(normal_vehicle_counts):.1f}")
        
        # Analyze distances
        collision_distances = [obs['closest_distance'] for obs in self.collision_observations 
                             if obs['closest_distance'] != float('inf')]
        normal_distances = [obs['closest_distance'] for obs in self.normal_observations 
                          if obs['closest_distance'] != float('inf')]
        
        if collision_distances:
            print(f"\n📏 DISTANCE ANALYSIS:")
            print(f"   Avg closest distance before collision: {np.mean(collision_distances):.1f}m")
            print(f"   Avg closest distance normally: {np.mean(normal_distances):.1f}m")
            
            if np.mean(collision_distances) > 5:
                print("   ⚠️ Agent can see nearby vehicles but still crashes - POLICY ISSUE")
            else:
                print("   🚨 Collisions happen with very close vehicles - REACTION TIME ISSUE")


def main():
    print("🔍 OBSERVATION DEBUGGER FOR RL AUTONOMOUS VEHICLE")
    print("=" * 60)
    
    # Try to find the latest trained model
    model_paths = [
        "experiments/results/models/PPO_debug_seed_0",
        "results/models/PPO_debug_seed_0",
        "experiments/results/models/PPO_seed_0",
        "results/models/PPO_seed_0"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    
    if model_path:
        print(f"✅ Found trained model: {model_path}")
    else:
        print("⚠️ No trained model found - will use random actions")
        print("   Train a model first: python experiments/train_agents.py")
    
    # Initialize debugger
    debugger = ObservationDebugger(model_path=model_path, use_custom_env=True)
    
    print("\n🎯 DEBUGGING MODES:")
    print("1. Live observation analysis during episodes")
    print("2. Pattern analysis of collision vs normal observations")
    
    mode = input("\nSelect mode (1 or 2): ").strip()
    
    if mode == "1":
        print("\n🔍 Starting live observation debugging...")
        print("Watch carefully what the agent 'sees' before collisions!")
        debugger.run_debugging_session()
        
        # Offer pattern analysis after
        if len(debugger.collision_observations) > 0:
            analyze = input("\n📊 Analyze collision patterns? (y/n): ").strip().lower()
            if analyze == 'y':
                debugger.analyze_collision_patterns()
    
    elif mode == "2":
        print("\n📊 Running automated pattern analysis...")
        # Run silently to collect data
        debugger.run_debugging_session()
        debugger.analyze_collision_patterns()
    
    else:
        print("Invalid mode selected")
    
    print("\n✅ Debugging session completed!")
    print("\n💡 NEXT STEPS based on findings:")
    print("   • If agent sees vehicles but crashes → Improve policy (more training, better rewards)")
    print("   • If agent doesn't see vehicles → Improve observation space or sensor range")
    print("   • If reaction time too slow → Increase training timesteps or learning rate")


if __name__ == "__main__":
    main()
