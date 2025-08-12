#!/usr/bin/env python3
"""
🎯 Q-LEARNING AGENT FOR AUTONOMOUS VEHICLE ROUNDABOUT NAVIGATION 🎯

This module implements a tabular Q-Learning agent for comparison with PPO.
It uses state discretization to handle the continuous observation space while
maintaining the same action space and evaluation metrics as the PPO agent.

🔍 KEY FEATURES:
- State space discretization for continuous observations
- Same 5-action discrete action space as PPO
- Epsilon-greedy exploration with decay
- Q-table persistence and loading
- Compatible evaluation metrics with PPO
- Safety-focused reward integration

📊 COMPARISON FRAMEWORK:
- Same environment (custom roundabout-v0)
- Same actions: [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
- Same evaluation: success rate, collision rate, episode rewards
- Same stopping criteria: safety-based performance
"""

import numpy as np
import pickle
import os
from collections import defaultdict
import json
from datetime import datetime
import gymnasium as gym
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class StateDiscretizer:
    """Converts continuous observations to discrete states for Q-Learning"""
    
    def __init__(self, n_vehicles_bins=3, distance_bins=5, speed_bins=4, 
                 angle_bins=8, position_bins=4):
        """
        Initialize state discretization parameters
        
        Args:
            n_vehicles_bins: Number of bins for vehicle count (0-2, 3-5, 6+)
            distance_bins: Number of bins for distances (very close, close, medium, far, very far)
            speed_bins: Number of bins for speeds (stopped, slow, medium, fast)
            angle_bins: Number of bins for angles (8 compass directions)
            position_bins: Number of bins for positions (quadrants)
        """
        self.n_vehicles_bins = n_vehicles_bins
        self.distance_bins = distance_bins
        self.speed_bins = speed_bins
        self.angle_bins = angle_bins
        self.position_bins = position_bins
        
        # Define bin boundaries
        self.distance_boundaries = [0, 3, 8, 15, 25, np.inf]  # meters
        self.speed_boundaries = [0, 2, 8, 15, np.inf]  # m/s
        self.vehicle_count_boundaries = [0, 2, 5, np.inf]
        
        print(f"🔢 State Discretizer initialized:")
        print(f"   Distance bins: {distance_bins} (boundaries: {self.distance_boundaries[:-1]})")
        print(f"   Speed bins: {speed_bins} (boundaries: {self.speed_boundaries[:-1]})")
        print(f"   Vehicle count bins: {n_vehicles_bins}")
        print(f"   Total possible states: ~{self._estimate_state_space_size()}")
    
    def _estimate_state_space_size(self):
        """Estimate the total number of possible discrete states"""
        return (self.n_vehicles_bins * 
                self.distance_bins * 
                self.speed_bins * 
                self.angle_bins * 
                self.position_bins * 
                self.speed_bins)  # ego speed
    
    def discretize_observation(self, obs: np.ndarray) -> Tuple[int, ...]:
        """
        Convert continuous observation to discrete state
        
        Args:
            obs: Continuous observation from environment (105-dim flattened array)
            
        Returns:
            Discrete state tuple
        """
        # Reshape observation to vehicle format (15 vehicles x 7 features)
        obs_reshaped = obs.reshape(-1, 7)
        ego_vehicle = obs_reshaped[0]
        other_vehicles = obs_reshaped[1:]
        
        # Extract ego vehicle information
        ego_x, ego_y = ego_vehicle[1], ego_vehicle[2]
        ego_vx, ego_vy = ego_vehicle[3], ego_vehicle[4]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        ego_heading = np.arctan2(ego_vehicle[6], ego_vehicle[5])  # sin, cos
        
        # Count nearby vehicles and find closest
        nearby_vehicles = []
        for vehicle in other_vehicles:
            if vehicle[0] > 0.5:  # Vehicle present
                distance = np.sqrt(vehicle[1]**2 + vehicle[2]**2)
                speed = np.sqrt(vehicle[3]**2 + vehicle[4]**2)
                nearby_vehicles.append((distance, speed))
        
        # Discretize vehicle count
        n_vehicles = len(nearby_vehicles)
        vehicle_count_bin = self._discretize_value(n_vehicles, self.vehicle_count_boundaries)
        
        # Discretize closest vehicle distance and speed
        if nearby_vehicles:
            closest_distance = min(vehicle[0] for vehicle in nearby_vehicles)
            avg_nearby_speed = np.mean([vehicle[1] for vehicle in nearby_vehicles])
        else:
            closest_distance = 50.0  # Large distance if no vehicles
            avg_nearby_speed = 0.0
        
        distance_bin = self._discretize_value(closest_distance, self.distance_boundaries)
        nearby_speed_bin = self._discretize_value(avg_nearby_speed, self.speed_boundaries)
        
        # Discretize ego vehicle state
        ego_speed_bin = self._discretize_value(ego_speed, self.speed_boundaries)
        
        # Discretize ego heading (8 compass directions)
        heading_bin = int((ego_heading + np.pi) / (2 * np.pi) * self.angle_bins) % self.angle_bins
        
        # Discretize ego position (quadrants relative to roundabout center)
        position_bin = self._discretize_position(ego_x, ego_y)
        
        # Create state tuple
        discrete_state = (
            vehicle_count_bin,
            distance_bin,
            nearby_speed_bin,
            ego_speed_bin,
            heading_bin,
            position_bin
        )
        
        return discrete_state
    
    def _discretize_value(self, value: float, boundaries: List[float]) -> int:
        """Discretize a continuous value using boundaries"""
        for i, boundary in enumerate(boundaries[1:]):
            if value < boundary:
                return i
        return len(boundaries) - 2  # Last bin
    
    def _discretize_position(self, x: float, y: float) -> int:
        """Discretize position into quadrants (0: NE, 1: NW, 2: SW, 3: SE)"""
        if x >= 0 and y >= 0:
            return 0  # Northeast
        elif x < 0 and y >= 0:
            return 1  # Northwest
        elif x < 0 and y < 0:
            return 2  # Southwest
        else:
            return 3  # Southeast


class QLearningAgent:
    """Tabular Q-Learning agent for autonomous vehicle navigation"""
    
    def __init__(self, action_space_size=5, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 state_discretizer=None):
        """
        Initialize Q-Learning agent
        
        Args:
            action_space_size: Number of discrete actions (5 for highway-env)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            state_discretizer: StateDiscretizer instance
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table using defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # State discretizer
        self.discretizer = state_discretizer or StateDiscretizer()
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'q_table_size': 0,
            'exploration_rate': epsilon,
            'avg_reward_window': [],
            'success_rate_window': [],
            'collision_rate_window': []
        }
        
        # Action names for debugging
        self.action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
        
        print(f"🎯 Q-Learning Agent initialized:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Discount factor: {discount_factor}")
        print(f"   Initial epsilon: {epsilon}")
        print(f"   Action space: {self.action_names}")
    
    def get_action(self, observation: np.ndarray, training=True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            observation: Continuous observation from environment
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action (0-4)
        """
        discrete_state = self.discretizer.discretize_observation(observation)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.action_space_size)
        else:
            # Exploit: best known action
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)
        
        return action
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """
        Update Q-table using Q-Learning update rule
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode is done
        """
        current_state = self.discretizer.discretize_observation(state)
        next_discrete_state = self.discretizer.discretize_observation(next_state)
        
        # Current Q-value
        current_q = self.q_table[current_state][action]
        
        # Next state max Q-value (0 if terminal state)
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_discrete_state])
        
        # Q-Learning update
        target_q = reward + self.discount_factor * next_max_q
        updated_q = current_q + self.learning_rate * (target_q - current_q)
        
        # Update Q-table
        self.q_table[current_state][action] = updated_q
        
        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['q_table_size'] = len(self.q_table)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats['exploration_rate'] = self.epsilon
    
    def update_episode_stats(self, episode_reward: float, success: bool, collision: bool):
        """Update episode-level statistics"""
        self.training_stats['episodes'] += 1
        
        # Maintain sliding windows for metrics
        window_size = 100
        
        self.training_stats['avg_reward_window'].append(episode_reward)
        if len(self.training_stats['avg_reward_window']) > window_size:
            self.training_stats['avg_reward_window'].pop(0)
        
        self.training_stats['success_rate_window'].append(1 if success else 0)
        if len(self.training_stats['success_rate_window']) > window_size:
            self.training_stats['success_rate_window'].pop(0)
        
        self.training_stats['collision_rate_window'].append(1 if collision else 0)
        if len(self.training_stats['collision_rate_window']) > window_size:
            self.training_stats['collision_rate_window'].pop(0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        stats = self.training_stats
        
        metrics = {
            'episodes': stats['episodes'],
            'total_steps': stats['total_steps'],
            'q_table_size': stats['q_table_size'],
            'exploration_rate': stats['exploration_rate'],
            'avg_reward': np.mean(stats['avg_reward_window']) if stats['avg_reward_window'] else 0,
            'success_rate': np.mean(stats['success_rate_window']) if stats['success_rate_window'] else 0,
            'collision_rate': np.mean(stats['collision_rate_window']) if stats['collision_rate_window'] else 0
        }
        
        return metrics
    
    def save_agent(self, filepath: str):
        """Save Q-table and agent parameters"""
        agent_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'training_stats': self.training_stats,
            'discretizer_params': {
                'n_vehicles_bins': self.discretizer.n_vehicles_bins,
                'distance_bins': self.discretizer.distance_bins,
                'speed_bins': self.discretizer.speed_bins,
                'angle_bins': self.discretizer.angle_bins,
                'position_bins': self.discretizer.position_bins
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"💾 Q-Learning agent saved to: {filepath}")
        print(f"   Q-table size: {len(self.q_table)} states")
        print(f"   Total training steps: {self.training_stats['total_steps']}")
    
    @classmethod
    def load_agent(cls, filepath: str):
        """Load Q-table and agent parameters"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Recreate discretizer
        disc_params = agent_data['discretizer_params']
        discretizer = StateDiscretizer(**disc_params)
        
        # Create agent
        agent = cls(
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor'],
            epsilon=agent_data['epsilon'],
            epsilon_decay=agent_data['epsilon_decay'],
            epsilon_min=agent_data['epsilon_min'],
            state_discretizer=discretizer
        )
        
        # Load Q-table
        agent.q_table = defaultdict(lambda: np.zeros(agent.action_space_size))
        for state, q_values in agent_data['q_table'].items():
            agent.q_table[state] = np.array(q_values)
        
        # Load training stats
        agent.training_stats = agent_data['training_stats']
        
        print(f"✅ Q-Learning agent loaded from: {filepath}")
        print(f"   Q-table size: {len(agent.q_table)} states")
        print(f"   Training episodes: {agent.training_stats['episodes']}")
        
        return agent
    
    def get_q_table_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the Q-table"""
        if not self.q_table:
            return {'size': 0, 'states_visited': 0}
        
        q_values = []
        for state_q_values in self.q_table.values():
            q_values.extend(state_q_values)
        
        return {
            'size': len(self.q_table),
            'states_visited': len(self.q_table),
            'total_q_values': len(q_values),
            'q_value_stats': {
                'mean': np.mean(q_values),
                'std': np.std(q_values),
                'min': np.min(q_values),
                'max': np.max(q_values)
            }
        }
    
    def print_training_progress(self, episode: int, episode_reward: float, 
                              success: bool, collision: bool):
        """Print training progress information"""
        metrics = self.get_performance_metrics()
        
        status = "✅ SUCCESS" if success else "💥 COLLISION" if collision else "⏱️ TIMEOUT"
        
        if episode % 100 == 0 or success or collision:
            print(f"\n📊 Episode {episode} - {status}")
            print(f"   Episode Reward: {episode_reward:.2f}")
            print(f"   Avg Reward (last 100): {metrics['avg_reward']:.2f}")
            print(f"   Success Rate: {metrics['success_rate']:.1%}")
            print(f"   Collision Rate: {metrics['collision_rate']:.1%}")
            print(f"   Exploration Rate: {metrics['exploration_rate']:.3f}")
            print(f"   Q-table Size: {metrics['q_table_size']} states")


# Factory function for easy integration
def create_q_learning_agent(learning_rate=0.1, discount_factor=0.99, 
                           epsilon=1.0, epsilon_decay=0.995) -> QLearningAgent:
    """
    Create a Q-Learning agent with default parameters optimized for roundabout navigation
    
    Args:
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma) 
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        
    Returns:
        QLearningAgent instance
    """
    discretizer = StateDiscretizer(
        n_vehicles_bins=3,    # Few, some, many vehicles
        distance_bins=5,      # Very close to very far
        speed_bins=4,         # Stopped, slow, medium, fast
        angle_bins=8,         # 8 compass directions
        position_bins=4       # 4 quadrants
    )
    
    return QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        state_discretizer=discretizer
    )


if __name__ == "__main__":
    # Test the Q-Learning agent
    print("🧪 Testing Q-Learning Agent")
    print("=" * 40)
    
    # Create agent
    agent = create_q_learning_agent()
    
    # Test state discretization
    print("\n🔢 Testing state discretization...")
    dummy_obs = np.random.random(105)  # Random observation
    discrete_state = agent.discretizer.discretize_observation(dummy_obs)
    print(f"Discrete state: {discrete_state}")
    
    # Test action selection
    print("\n🎮 Testing action selection...")
    action = agent.get_action(dummy_obs)
    print(f"Selected action: {agent.action_names[action]} ({action})")
    
    # Test Q-table update
    print("\n📚 Testing Q-table update...")
    agent.update_q_table(dummy_obs, action, 1.0, dummy_obs, False)
    print(f"Q-table size after update: {len(agent.q_table)}")
    
    print("\n✅ Q-Learning agent test completed!")
