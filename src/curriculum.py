#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING FOR AUTONOMOUS VEHICLE RL TRAINING 🎓

This module implements curriculum learning to gradually increase the difficulty
of driving scenarios. Starting with easier scenarios and progressively adding
complexity helps the agent learn robust collision avoidance more effectively.

🎯 CURRICULUM STAGES:
1. BEGINNER: Low traffic density, slower vehicles, wider lanes
2. INTERMEDIATE: Normal traffic, mixed speeds, some aggressive drivers  
3. ADVANCED: High traffic density, fast vehicles, narrow margins
4. EXPERT: Rush hour conditions, unpredictable behaviors, stress testing

🧠 WHY THIS WORKS:
- Prevents the agent from getting stuck in local optima
- Builds foundational skills before tackling complex scenarios
- Reduces training time by focusing on progressive skill building
- Improves final performance and generalization

📈 AUTOMATIC PROGRESSION:
- Progresses based on success rate and collision metrics
- Allows manual override for research purposes
- Supports fine-grained difficulty adjustment
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple
import warnings

class CurriculumManager:
    """Manages progressive difficulty in RL training"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config.copy()
        self.current_level = 0
        self.level_names = ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"]
        
        # Performance tracking
        self.level_episodes = 0
        self.level_successes = 0
        self.level_collisions = 0
        self.level_rewards = []
        
        # Progression criteria
        self.min_episodes_per_level = 20
        self.success_threshold = 0.8  # 80% success rate to advance
        self.max_collision_rate = 0.2  # Max 20% collision rate
        
        print(f"🎓 Curriculum Learning Initialized")
        print(f"   Starting at level: {self.level_names[0]}")
        print(f"   Progression criteria: {self.success_threshold:.0%} success, <{self.max_collision_rate:.0%} collision")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get environment configuration for current curriculum level"""
        
        config = self.base_config.copy()
        
        if self.current_level == 0:  # BEGINNER
            config.update({
                'vehicles_count': 10,        # Low traffic
                'controlled_vehicles': 1,
                'initial_spacing': 3.0,      # More space between vehicles
                'spawn_probability': 0.5,    # Fewer random spawns
                'duration': 150,             # Shorter episodes
                'simulation_frequency': 10,   # Slower simulation
                'policy_frequency': 5,       # More reaction time
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'speed_limit': 15,           # Slower traffic
                'reward_speed_range': [10, 15],
                'normalize_reward': False
            })
            
        elif self.current_level == 1:  # INTERMEDIATE  
            config.update({
                'vehicles_count': 15,        # Normal traffic
                'controlled_vehicles': 1,
                'initial_spacing': 2.5,      # Normal spacing
                'spawn_probability': 0.7,    # Normal spawns
                'duration': 200,             # Normal episodes
                'simulation_frequency': 15,   # Normal speed
                'policy_frequency': 5,
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'speed_limit': 20,           # Normal speed
                'reward_speed_range': [15, 20],
                'normalize_reward': False
            })
            
        elif self.current_level == 2:  # ADVANCED
            config.update({
                'vehicles_count': 20,        # High traffic
                'controlled_vehicles': 1, 
                'initial_spacing': 2.0,      # Tight spacing
                'spawn_probability': 0.9,    # Frequent spawns
                'duration': 250,             # Longer episodes
                'simulation_frequency': 15,
                'policy_frequency': 5,
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'speed_limit': 25,           # Fast traffic
                'reward_speed_range': [20, 25],
                'normalize_reward': False
            })
            
        else:  # EXPERT (level 3+)
            config.update({
                'vehicles_count': 25,        # Very high traffic
                'controlled_vehicles': 1,
                'initial_spacing': 1.5,      # Very tight spacing
                'spawn_probability': 1.0,    # Maximum spawns
                'duration': 300,             # Long episodes
                'simulation_frequency': 15,
                'policy_frequency': 5, 
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'speed_limit': 30,           # Very fast traffic
                'reward_speed_range': [25, 30],
                'normalize_reward': False
            })
        
        return config
    
    def record_episode_result(self, info: Dict[str, Any], reward: float) -> bool:
        """Record episode result and check if ready to advance level"""
        
        self.level_episodes += 1
        self.level_rewards.append(reward)
        
        # Track success/collision
        if info.get('arrived', False):
            self.level_successes += 1
        elif info.get('crashed', False):
            self.level_collisions += 1
        
        # Check if ready to advance
        if self.level_episodes >= self.min_episodes_per_level:
            success_rate = self.level_successes / self.level_episodes
            collision_rate = self.level_collisions / self.level_episodes
            avg_reward = np.mean(self.level_rewards)
            
            print(f"\n📊 Level {self.level_names[self.current_level]} Performance:")
            print(f"   Episodes: {self.level_episodes}")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Collision Rate: {collision_rate:.1%}")
            print(f"   Average Reward: {avg_reward:.2f}")
            
            # Check advancement criteria
            if (success_rate >= self.success_threshold and 
                collision_rate <= self.max_collision_rate and
                self.current_level < len(self.level_names) - 1):
                
                print(f"✅ ADVANCING to {self.level_names[self.current_level + 1]}!")
                self._advance_level()
                return True
            elif self.current_level == len(self.level_names) - 1:
                print(f"🏆 EXPERT level mastered! Training complete.")
                return False
            else:
                print(f"📈 Continue training at {self.level_names[self.current_level]} level")
                return False
        
        return False
    
    def _advance_level(self):
        """Advance to next curriculum level"""
        self.current_level += 1
        
        # Reset tracking for new level
        self.level_episodes = 0
        self.level_successes = 0
        self.level_collisions = 0
        self.level_rewards = []
        
        print(f"🎓 Now training at {self.level_names[self.current_level]} level")
    
    def get_level_info(self) -> Dict[str, Any]:
        """Get current level information"""
        return {
            'level': self.current_level,
            'level_name': self.level_names[self.current_level],
            'episodes': self.level_episodes,
            'successes': self.level_successes,
            'collisions': self.level_collisions,
            'progress': min(self.level_episodes / self.min_episodes_per_level, 1.0)
        }
    
    def force_advance(self):
        """Manually advance to next level (for testing/research)"""
        if self.current_level < len(self.level_names) - 1:
            print(f"🔧 MANUAL ADVANCE: {self.level_names[self.current_level]} → {self.level_names[self.current_level + 1]}")
            self._advance_level()
        else:
            print("🏆 Already at maximum level (EXPERT)")


class CurriculumEnvironment:
    """Wrapper that applies curriculum learning to highway environments"""
    
    def __init__(self, base_env_id: str = "roundabout-v0", curriculum_config: Dict = None):
        self.base_env_id = base_env_id
        self.curriculum = CurriculumManager(curriculum_config or {})
        self.env = None
        self._create_environment()
    
    def _create_environment(self):
        """Create environment with current curriculum configuration"""
        
        config = self.curriculum.get_current_config()
        
        # Create environment
        self.env = gym.make(self.base_env_id)
        self.env.configure(config)
        
        level_info = self.curriculum.get_level_info()
        print(f"🎯 Environment configured for {level_info['level_name']} level")
    
    def reset(self, **kwargs):
        """Reset environment, potentially updating curriculum"""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step environment and track curriculum progress"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # If episode ended, record result and check advancement
        if done or truncated:
            advanced = self.curriculum.record_episode_result(info, reward)
            if advanced:
                # Recreate environment with new curriculum level
                self.env.close()
                self._create_environment()
        
        return obs, reward, done, truncated, info
    
    def close(self):
        """Close environment"""
        if self.env:
            self.env.close()
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying environment"""
        return getattr(self.env, name)


# Factory function for easy integration
def create_curriculum_environment(env_id: str = "roundabout-v0", 
                                custom_env: bool = False) -> CurriculumEnvironment:
    """
    Create a curriculum learning environment
    
    Args:
        env_id: Base environment ID
        custom_env: Whether to use custom reward environment
        
    Returns:
        CurriculumEnvironment instance
    """
    
    if custom_env:
        # Use with custom environment
        from src.environment import register_custom_env
        register_custom_env()
        env_id = "custom-roundabout-v0"
    
    # Default curriculum configuration
    base_config = {
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 15,
            'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            'features_range': {
                'x': [-100, 100],
                'y': [-100, 100], 
                'vx': [-20, 20],
                'vy': [-20, 20]
            },
            'absolute': False,
            'flatten': True,
            'observe_intentions': False
        },
        'action': {
            'type': 'DiscreteMetaAction',
        },
        'simulation_frequency': 15,
        'policy_frequency': 5,
        'duration': 200,
        'collision_reward': -1,
        'high_speed_reward': 0.2,
        'right_lane_reward': 0,
        'lane_change_reward': 0,
        'reward_speed_range': [20, 30],
        'normalize_reward': False,
        'offroad_terminal': False
    }
    
    return CurriculumEnvironment(env_id, base_config)


# Example usage for integration with training scripts
def integrate_curriculum_with_training():
    """Example of how to integrate curriculum learning with existing training"""
    
    print("🎓 CURRICULUM LEARNING INTEGRATION EXAMPLE")
    print("=" * 50)
    
    # Create curriculum environment
    curriculum_env = create_curriculum_environment("roundabout-v0", custom_env=True)
    
    # Training loop example
    for episode in range(100):
        obs, info = curriculum_env.reset()
        episode_reward = 0
        
        for step in range(300):
            # Get action from your trained model
            action = curriculum_env.action_space.sample()  # Replace with model.predict()
            
            obs, reward, done, truncated, info = curriculum_env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        # Curriculum automatically tracks progress and advances levels
        level_info = curriculum_env.curriculum.get_level_info()
        print(f"Episode {episode}: {level_info['level_name']} level, reward: {episode_reward:.2f}")
    
    curriculum_env.close()


if __name__ == "__main__":
    integrate_curriculum_with_training()
