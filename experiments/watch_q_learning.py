#!/usr/bin/env python3
"""
👀 WATCH Q-LEARNING AGENT PERFORMANCE 👀

This script loads and visualizes a trained Q-Learning agent's performance
in the roundabout environment. Equivalent to watch_agents.py for PPO.

🎯 FEATURES:
- Load trained Q-Learning models
- Visualize agent performance with rendering
- Display action decisions and state information
- Compare different Q-Learning checkpoints
- Performance metrics calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import time
from src.q_learning_agent import QLearningAgent
from src.environment import register_custom_env


def watch_q_learning_agent(model_path, n_episodes=5, custom_env=True, 
                          episode_delay=1.0, step_delay=0.1):
    """
    Watch Q-Learning agent performance
    
    Args:
        model_path: Path to saved Q-Learning model (.pkl file)
        n_episodes: Number of episodes to watch
        custom_env: Whether to use custom environment
        episode_delay: Delay between episodes (seconds)
        step_delay: Delay between steps (seconds)
    """
    
    print(f"👀 WATCHING Q-LEARNING AGENT PERFORMANCE")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Custom environment: {custom_env}")
    print("=" * 50)
    
    # Create environment
    if custom_env:
        register_custom_env()
        env = gym.make('custom-roundabout-v0', render_mode="human")
        print("🎁 Using CUSTOM environment")
    else:
        env = gym.make('roundabout-v0', render_mode="human")
        print("⚠️ Using STANDARD environment")
    
    # Load Q-Learning agent
    try:
        agent = QLearningAgent.load_agent(model_path)
        print(f"✅ Loaded Q-Learning agent")
        print(f"   Q-table size: {len(agent.q_table)} states")
        print(f"   Training episodes: {agent.training_stats['episodes']}")
        print(f"   Exploration rate: {agent.epsilon:.3f}")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return
    
    # Performance tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    
    action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    action_counts = {name: 0 for name in action_names}
    
    # Watch episodes
    for episode in range(n_episodes):
        print(f"\n🎬 EPISODE {episode + 1}/{n_episodes}")
        print("-" * 30)
        
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = truncated = False
        episode_actions = []
        
        while not (done or truncated):
            # Get agent's action (no exploration)
            action = agent.get_action(obs, training=False)
            action_name = action_names[action]
            episode_actions.append(action)
            action_counts[action_name] += 1
            
            # Display current state info
            discrete_state = agent.discretizer.discretize_observation(obs)
            q_values = agent.q_table[discrete_state]
            max_q_value = np.max(q_values)
            
            if steps % 20 == 0:  # Display every 20 steps
                print(f"  Step {steps:3d}: {action_name:10s} | "
                      f"Q-value: {q_values[action]:.2f} | "
                      f"Max Q: {max_q_value:.2f} | "
                      f"State: {discrete_state}")
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Delay for visualization
            time.sleep(step_delay)
        
        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        outcome = "❓ UNKNOWN"
        if info.get('crashed', False):
            collision_count += 1
            outcome = "💥 COLLISION"
        elif info.get('arrived', False):
            success_count += 1
            outcome = "✅ SUCCESS"
        else:
            outcome = "⏱️ TIMEOUT"
        
        print(f"\n📊 Episode {episode + 1} Results:")
        print(f"   Outcome: {outcome}")
        print(f"   Steps: {steps}")
        print(f"   Reward: {episode_reward:.2f}")
        print(f"   Actions taken: {len(set(episode_actions))} unique")
        
        # Most common actions this episode
        episode_action_names = [action_names[a] for a in episode_actions]
        from collections import Counter
        common_actions = Counter(episode_action_names).most_common(3)
        print(f"   Top actions: {', '.join([f'{action}({count})' for action, count in common_actions])}")
        
        if episode < n_episodes - 1:
            print(f"\n⏸️ Episode delay ({episode_delay}s)...")
            time.sleep(episode_delay)
    
    # Overall performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY ({n_episodes} episodes)")
    print("=" * 50)
    print(f"Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes:.1%})")
    print(f"Collision Rate: {collision_count}/{n_episodes} ({collision_count/n_episodes:.1%})")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    
    print(f"\n🎮 ACTION DISTRIBUTION:")
    total_actions = sum(action_counts.values())
    for action_name, count in action_counts.items():
        percentage = count / total_actions * 100 if total_actions > 0 else 0
        print(f"   {action_name:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n🧠 Q-TABLE STATISTICS:")
    q_summary = agent.get_q_table_summary()
    print(f"   States visited: {q_summary['states_visited']}")
    print(f"   Q-value range: [{q_summary['q_value_stats']['min']:.2f}, {q_summary['q_value_stats']['max']:.2f}]")
    print(f"   Q-value mean: {q_summary['q_value_stats']['mean']:.2f}")
    
    # Performance comparison hints
    print(f"\n💡 PERFORMANCE ANALYSIS:")
    if success_count / n_episodes >= 0.8:
        print("   ✅ Strong performance - agent completes routes reliably")
    elif success_count / n_episodes >= 0.5:
        print("   ⚠️ Moderate performance - some room for improvement")
    else:
        print("   🚨 Weak performance - needs more training or hyperparameter tuning")
    
    if collision_count / n_episodes <= 0.1:
        print("   ✅ Safe driving - low collision rate")
    elif collision_count / n_episodes <= 0.3:
        print("   ⚠️ Moderate safety - some collisions occurring")
    else:
        print("   🚨 Safety concerns - high collision rate")
    
    avg_reward = np.mean(episode_rewards)
    if avg_reward >= 10:
        print("   ✅ High rewards - efficient and successful navigation")
    elif avg_reward >= 0:
        print("   ⚠️ Moderate rewards - acceptable but could be optimized")
    else:
        print("   🚨 Low/negative rewards - poor performance")
    
    env.close()
    
    return {
        'success_rate': success_count / n_episodes,
        'collision_rate': collision_count / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'action_distribution': action_counts,
        'q_table_stats': q_summary
    }


def main():
    print("👀 Q-LEARNING AGENT PERFORMANCE VIEWER")
    print("=" * 50)
    
    # Find available Q-Learning models
    model_paths = [
        "experiments/results/models/Q_Learning_final_seed_0.pkl",
        "experiments/results/models/Q_Learning_best_seed_0.pkl"
    ]
    
    available_models = []
    for path in model_paths:
        if os.path.exists(path):
            available_models.append(path)
    
    if not available_models:
        print("❌ No trained Q-Learning models found!")
        print("   Train a model first: python experiments/train_q_learning.py")
        print("   Expected locations:")
        for path in model_paths:
            print(f"   • {path}")
        return
    
    print("✅ Available Q-Learning models:")
    for i, path in enumerate(available_models):
        model_name = os.path.basename(path).replace('.pkl', '')
        print(f"   {i+1}. {model_name}")
    
    # Model selection
    if len(available_models) == 1:
        selected_model = available_models[0]
        print(f"\nUsing: {os.path.basename(selected_model)}")
    else:
        try:
            choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
            selected_model = available_models[choice]
        except (ValueError, IndexError):
            print("Invalid selection, using first model")
            selected_model = available_models[0]
    
    # Configuration
    n_episodes = 5
    custom_env = True
    
    print(f"\n🎬 VIEWING CONFIGURATION:")
    print(f"   Model: {os.path.basename(selected_model)}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Environment: {'Custom' if custom_env else 'Standard'}")
    
    input("\n🎮 Press Enter to start watching the Q-Learning agent...")
    
    # Watch the agent
    performance = watch_q_learning_agent(
        model_path=selected_model,
        n_episodes=n_episodes,
        custom_env=custom_env,
        episode_delay=2.0,
        step_delay=0.1
    )
    
    print(f"\n✅ Viewing completed!")
    print(f"\n🔬 QUICK COMPARISON HINTS:")
    print(f"   Compare these metrics with your PPO agent:")
    print(f"   • Success rate: {performance['success_rate']:.1%}")
    print(f"   • Collision rate: {performance['collision_rate']:.1%}")
    print(f"   • Average reward: {performance['mean_reward']:.2f}")
    print(f"   • Average episode length: {performance['mean_length']:.1f} steps")
    print(f"\n📊 For detailed comparison, run:")
    print(f"   python experiments/compare_algorithms.py")


if __name__ == "__main__":
    main()
