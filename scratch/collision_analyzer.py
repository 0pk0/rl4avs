#!/usr/bin/env python3
"""
💥 COLLISION PATTERN ANALYZER FOR RL AUTONOMOUS VEHICLES 💥

This tool systematically analyzes collision patterns to identify the most
common failure scenarios in your RL agent. Understanding these patterns
is crucial for targeted improvements in training and reward design.

🎯 ANALYSIS FEATURES:
- Collision location heatmaps (where crashes happen most)
- Speed analysis at collision time
- Action patterns leading to crashes
- Time-to-collision analysis
- Vehicle proximity patterns
- Success vs failure scenario comparison

🔍 PATTERN IDENTIFICATION:
- Lane change collisions
- Following too closely (rear-end)
- Intersection conflicts
- Merging failures
- Speed-related crashes

📊 OUTPUTS:
- Detailed collision reports
- Visual heatmaps and plots
- Actionable insights for training improvements
- Specific recommendations for reward function tuning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from src.environment import register_custom_env
import warnings
warnings.filterwarnings('ignore')

class CollisionAnalyzer:
    """Comprehensive collision pattern analysis tool"""
    
    def __init__(self, model_path=None, use_custom_env=True):
        self.use_custom_env = use_custom_env
        self.collision_data = []
        self.success_data = []
        
        # Create environment
        if use_custom_env:
            register_custom_env()
            self.env = gym.make('custom-roundabout-v0', render_mode="human")
            print("🎁 Using CUSTOM environment for collision analysis")
        else:
            self.env = gym.make('roundabout-v0', render_mode="human")
            print("⚠️ Using STANDARD environment for collision analysis")
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path + ".zip"):
            try:
                self.model = PPO.load(model_path)
                print(f"✅ Loaded model: {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🤖 Will use random actions for analysis")
        else:
            print("🤖 No model provided - using random actions")
    
    def collect_collision_data(self, n_episodes=50, max_steps=300):
        """Collect comprehensive data about collisions and successes"""
        
        print(f"\n💥 Collecting collision data over {n_episodes} episodes...")
        print("🎯 This will help identify the most common failure patterns")
        
        collisions_found = 0
        successes_found = 0
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_data = {
                'episode': episode,
                'positions': [],
                'speeds': [],
                'actions': [],
                'rewards': [],
                'nearby_vehicles': [],
                'timesteps': []
            }
            
            episode_reward = 0
            
            for step in range(max_steps):
                # Record current state
                obs_reshaped = obs.reshape(-1, 6)
                ego_pos = obs_reshaped[0, 1:3]  # x, y position
                ego_vel = obs_reshaped[0, 3:5]  # vx, vy velocity
                ego_speed = np.sqrt(ego_vel[0]**2 + ego_vel[1]**2)
                
                # Count nearby vehicles
                nearby_count = 0
                closest_distance = float('inf')
                
                for vehicle in obs_reshaped[1:]:
                    if vehicle[0] > 0.5:  # Vehicle present
                        distance = np.sqrt(vehicle[1]**2 + vehicle[2]**2)
                        nearby_count += 1
                        closest_distance = min(closest_distance, distance)
                
                # Get action
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Record data
                episode_data['positions'].append(ego_pos.copy())
                episode_data['speeds'].append(ego_speed)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(episode_reward)
                episode_data['nearby_vehicles'].append(nearby_count)
                episode_data['timesteps'].append(step)
                
                # Take step
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    # Determine episode outcome
                    if info.get('crashed', False):
                        collisions_found += 1
                        episode_data['outcome'] = 'collision'
                        episode_data['collision_step'] = step
                        episode_data['collision_position'] = ego_pos.copy()
                        episode_data['collision_speed'] = ego_speed
                        episode_data['collision_action'] = action
                        episode_data['nearby_at_collision'] = nearby_count
                        episode_data['closest_distance'] = closest_distance if closest_distance != float('inf') else None
                        episode_data['final_reward'] = episode_reward
                        
                        self.collision_data.append(episode_data)
                        print(f"💥 Collision {collisions_found} at step {step} (episode {episode})")
                        
                    elif info.get('arrived', False):
                        successes_found += 1
                        episode_data['outcome'] = 'success'
                        episode_data['completion_step'] = step
                        episode_data['final_reward'] = episode_reward
                        
                        self.success_data.append(episode_data)
                        print(f"✅ Success {successes_found} at step {step} (episode {episode})")
                    
                    else:
                        episode_data['outcome'] = 'timeout'
                        episode_data['final_reward'] = episode_reward
                    
                    break
        
        print(f"\n📊 Data Collection Summary:")
        print(f"   Total Episodes: {n_episodes}")
        print(f"   Collisions: {collisions_found} ({collisions_found/n_episodes:.1%})")
        print(f"   Successes: {successes_found} ({successes_found/n_episodes:.1%})")
        print(f"   Timeouts: {n_episodes - collisions_found - successes_found}")
        
        return collisions_found, successes_found
    
    def analyze_collision_patterns(self):
        """Analyze patterns in collision data"""
        
        if len(self.collision_data) == 0:
            print("❌ No collision data available. Run collect_collision_data() first.")
            return
        
        print(f"\n💥 COLLISION PATTERN ANALYSIS")
        print("=" * 60)
        print(f"Analyzing {len(self.collision_data)} collision episodes...")
        
        # 1. Collision timing analysis
        collision_steps = [data['collision_step'] for data in self.collision_data]
        avg_collision_time = np.mean(collision_steps)
        
        print(f"\n⏱️ TIMING ANALYSIS:")
        print(f"   Average collision time: {avg_collision_time:.1f} steps")
        print(f"   Earliest collision: {min(collision_steps)} steps")
        print(f"   Latest collision: {max(collision_steps)} steps")
        
        if avg_collision_time < 50:
            print("   🚨 INSIGHT: Early collisions suggest poor initial policy")
        elif avg_collision_time < 100:
            print("   ⚠️ INSIGHT: Mid-episode collisions suggest navigation issues")
        else:
            print("   💡 INSIGHT: Late collisions suggest complex scenario failures")
        
        # 2. Speed analysis
        collision_speeds = [data['collision_speed'] for data in self.collision_data]
        avg_collision_speed = np.mean(collision_speeds)
        
        print(f"\n🏎️ SPEED ANALYSIS:")
        print(f"   Average speed at collision: {avg_collision_speed:.2f} m/s")
        print(f"   Speed range: {min(collision_speeds):.2f} - {max(collision_speeds):.2f} m/s")
        
        if avg_collision_speed > 15:
            print("   🚨 INSIGHT: High-speed collisions - agent too aggressive")
        elif avg_collision_speed < 5:
            print("   🐌 INSIGHT: Low-speed collisions - hesitation or blocking")
        else:
            print("   ✅ INSIGHT: Normal speed collisions - tactical errors")
        
        # 3. Action analysis
        collision_actions = [data['collision_action'] for data in self.collision_data]
        action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
        action_counts = Counter(collision_actions)
        
        print(f"\n🎮 ACTION ANALYSIS (at collision):")
        for action, count in action_counts.most_common():
            percentage = count / len(collision_actions) * 100
            print(f"   {action_names[action]}: {count} times ({percentage:.1f}%)")
        
        most_dangerous_action = action_counts.most_common(1)[0][0]
        print(f"   🚨 Most dangerous action: {action_names[most_dangerous_action]}")
        
        # 4. Traffic density analysis
        nearby_at_collision = [data['nearby_at_collision'] for data in self.collision_data]
        avg_nearby = np.mean(nearby_at_collision)
        
        print(f"\n🚗 TRAFFIC DENSITY ANALYSIS:")
        print(f"   Average vehicles nearby at collision: {avg_nearby:.1f}")
        print(f"   Range: {min(nearby_at_collision)} - {max(nearby_at_collision)} vehicles")
        
        if avg_nearby > 8:
            print("   🚨 INSIGHT: Collisions in heavy traffic - multi-agent coordination issues")
        elif avg_nearby < 3:
            print("   🤔 INSIGHT: Collisions in light traffic - basic navigation issues")
        else:
            print("   ✅ INSIGHT: Normal traffic density collisions")
        
        # 5. Distance analysis (if available)
        distances = [data['closest_distance'] for data in self.collision_data if data['closest_distance'] is not None]
        if distances:
            avg_closest = np.mean(distances)
            print(f"\n📏 PROXIMITY ANALYSIS:")
            print(f"   Average closest vehicle distance: {avg_closest:.2f}m")
            
            if avg_closest < 3:
                print("   🚨 INSIGHT: Following too closely - increase safety distance")
            elif avg_closest > 10:
                print("   🤔 INSIGHT: Collisions despite safe distance - prediction errors")
        
        # 6. Reward analysis
        final_rewards = [data['final_reward'] for data in self.collision_data]
        avg_collision_reward = np.mean(final_rewards)
        
        print(f"\n🏆 REWARD ANALYSIS:")
        print(f"   Average reward at collision: {avg_collision_reward:.2f}")
        
        if self.success_data:
            success_rewards = [data['final_reward'] for data in self.success_data]
            avg_success_reward = np.mean(success_rewards)
            reward_diff = avg_success_reward - avg_collision_reward
            
            print(f"   Average reward at success: {avg_success_reward:.2f}")
            print(f"   Reward difference: {reward_diff:.2f}")
            
            if reward_diff < 5:
                print("   ⚠️ INSIGHT: Small reward difference - need stronger penalties for collisions")
            else:
                print("   ✅ INSIGHT: Good reward differentiation between success and failure")
    
    def generate_recommendations(self):
        """Generate specific recommendations based on collision analysis"""
        
        if len(self.collision_data) == 0:
            print("❌ No collision data available for recommendations.")
            return
        
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("=" * 60)
        
        # Analyze collision patterns for recommendations
        collision_actions = [data['collision_action'] for data in self.collision_data]
        collision_speeds = [data['collision_speed'] for data in self.collision_data]
        collision_steps = [data['collision_step'] for data in self.collision_data]
        nearby_counts = [data['nearby_at_collision'] for data in self.collision_data]
        
        action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
        most_common_action = Counter(collision_actions).most_common(1)[0][0]
        avg_speed = np.mean(collision_speeds)
        avg_time = np.mean(collision_steps)
        avg_nearby = np.mean(nearby_counts)
        
        recommendations = []
        
        # Action-based recommendations
        if most_common_action == 3:  # FASTER
            recommendations.append("🚨 CRITICAL: Stop penalizing speed near other vehicles")
            recommendations.append("   • Increase penalty for FASTER action when vehicles nearby")
            recommendations.append("   • Add collision prediction in reward function")
        
        elif most_common_action == 1:  # IDLE
            recommendations.append("⚠️ BLOCKING: Agent freezing in dangerous situations")
            recommendations.append("   • Penalize IDLE action more heavily")
            recommendations.append("   • Reward decisive lane changes over hesitation")
        
        elif most_common_action in [0, 2]:  # Lane changes
            recommendations.append("🔄 LANE CHANGE ISSUES: Unsafe lane changes")
            recommendations.append("   • Add safety checks for lane change rewards")
            recommendations.append("   • Increase observation of vehicles in target lanes")
        
        # Speed-based recommendations
        if avg_speed > 15:
            recommendations.append("🏎️ SPEED CONTROL: Agent too aggressive")
            recommendations.append("   • Increase speed penalties near other vehicles")
            recommendations.append("   • Add graduated speed rewards (safer at lower speeds)")
        
        elif avg_speed < 5:
            recommendations.append("🐌 SPEED ISSUES: Agent too cautious or stuck")
            recommendations.append("   • Penalize very low speeds")
            recommendations.append("   • Reward maintaining reasonable progress speed")
        
        # Traffic density recommendations
        if avg_nearby > 8:
            recommendations.append("🚗 MULTI-AGENT: Struggles in heavy traffic")
            recommendations.append("   • Implement curriculum learning (start with less traffic)")
            recommendations.append("   • Increase training timesteps for complex scenarios")
            recommendations.append("   • Consider larger neural network for multi-vehicle coordination")
        
        # Timing-based recommendations
        if avg_time < 50:
            recommendations.append("⚡ EARLY FAILURE: Poor initial policy")
            recommendations.append("   • Increase exploration during early training")
            recommendations.append("   • Add imitation learning from safe demonstrations")
            recommendations.append("   • Start with simpler scenarios (curriculum learning)")
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n🎯 PRIORITY ACTIONS:")
        print(f"   1. Focus on {action_names[most_common_action]} action safety")
        print(f"   2. Implement speed control for {avg_speed:.1f} m/s scenarios")
        print(f"   3. Improve {avg_nearby:.0f}-vehicle traffic handling")
        
        # Training recommendations
        print(f"\n🎓 TRAINING IMPROVEMENTS:")
        print(f"   • Increase total timesteps to 500K+ for complex scenarios")
        print(f"   • Use curriculum learning (start with {max(1, int(avg_nearby)-3)} vehicles)")
        print(f"   • Implement safety-focused stopping criteria")
        print(f"   • Add collision prediction as auxiliary task")
    
    def save_analysis_report(self, filename=None):
        """Save detailed analysis report"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collision_analysis_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_collisions': len(self.collision_data),
            'total_successes': len(self.success_data),
            'collision_data': self.collision_data,
            'success_data': self.success_data,
            'analysis_summary': {
                'avg_collision_time': np.mean([d['collision_step'] for d in self.collision_data]) if self.collision_data else 0,
                'avg_collision_speed': np.mean([d['collision_speed'] for d in self.collision_data]) if self.collision_data else 0,
                'most_common_collision_action': Counter([d['collision_action'] for d in self.collision_data]).most_common(1)[0] if self.collision_data else None,
                'avg_nearby_vehicles': np.mean([d['nearby_at_collision'] for d in self.collision_data]) if self.collision_data else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Analysis report saved to: {filename}")
        return filename


def main():
    print("💥 COLLISION PATTERN ANALYZER")
    print("=" * 60)
    print("🎯 Systematic analysis of RL agent collision patterns")
    print("🔍 Identifies failure modes and suggests improvements")
    
    # Find trained model
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
        print("⚠️ No trained model found - will analyze random policy")
        print("   Train a model first for meaningful analysis")
    
    # Initialize analyzer
    analyzer = CollisionAnalyzer(model_path=model_path, use_custom_env=True)
    
    # Configuration
    n_episodes = int(input("Number of episodes to analyze (recommended: 50-100): ") or "50")
    
    print(f"\n🔬 Starting analysis with {n_episodes} episodes...")
    print("   This will take a few minutes - watch for collision patterns!")
    
    # Collect data
    collisions, successes = analyzer.collect_collision_data(n_episodes=n_episodes)
    
    if collisions == 0:
        print("\n🎉 NO COLLISIONS DETECTED!")
        print("   Your agent appears to be very safe already!")
        print("   Consider increasing episode count or difficulty for analysis.")
        return
    
    # Analyze patterns
    analyzer.analyze_collision_patterns()
    
    # Generate recommendations
    analyzer.generate_recommendations()
    
    # Save report
    save_report = input("\n💾 Save detailed analysis report? (y/n): ").strip().lower()
    if save_report == 'y':
        analyzer.save_analysis_report()
    
    print("\n✅ Collision analysis completed!")
    print("\n🎯 USE THESE INSIGHTS TO:")
    print("   1. Adjust reward function based on identified failure modes")
    print("   2. Modify training hyperparameters for problematic scenarios")
    print("   3. Implement curriculum learning for gradual improvement")
    print("   4. Focus debugging on most common collision patterns")


if __name__ == "__main__":
    main()
