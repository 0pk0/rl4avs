#!/usr/bin/env python3
"""
🎯 Q-LEARNING TRAINING FOR AUTONOMOUS VEHICLE ROUNDABOUT NAVIGATION 🎯

This script trains a Q-Learning agent using the same environment, evaluation metrics,
and stopping criteria as the PPO agent for direct comparison.

🔧 FEATURES:
- Same environment (custom roundabout-v0) as PPO
- Same action space: [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
- Same evaluation metrics: success rate, collision rate, episode rewards
- Same safety-based stopping criteria
- Comparable training resources and time limits
- Detailed logging and progress tracking

📊 COMPARISON FRAMEWORK:
- Uses same episode limits and evaluation frequency as PPO
- Same custom reward function for consistency
- Saves models in comparable format for analysis
- Generates same performance metrics for comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the project root to sys.path if not already there, for resolving 'src' imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from datetime import datetime
from collections import deque
import wandb # Import wandb
import argparse # Import argparse

from src.q_learning_agent import create_q_learning_agent, QLearningAgent
from src.environment import register_custom_env, make_env
from stable_baselines3.common.evaluation import evaluate_policy


class QLearningExperimentRunner:
    """Enhanced experiment runner for Q-Learning with same interface as PPO runner"""
    
    def __init__(self, env_name="roundabout-v0", use_custom_env=True):
        self.env_name = env_name
        self.use_custom_env = use_custom_env
        self.results = {}
        
        # Ensure directories exist
        os.makedirs("experiments/results/models", exist_ok=True)
        os.makedirs("experiments/results/logs", exist_ok=True)
        
        print(f"🎯 Q-Learning Experiment Runner initialized")
        print(f"   Environment: {env_name}")
        print(f"   Custom rewards: {use_custom_env}")
    
    def create_environment(self, render_mode=None):
        """Create environment (same logic as PPO setup)"""
        if self.use_custom_env:
            register_custom_env()
            if render_mode:
                env = gym.make('custom-roundabout-v0', render_mode=render_mode)
            else:
                env = gym.make('custom-roundabout-v0')
            print("🎁 Using CUSTOM environment for Q-Learning")
        else:
            if render_mode:
                env = gym.make(self.env_name, render_mode=render_mode)
            else:
                env = gym.make(self.env_name)
            print("⚠️ Using STANDARD environment for Q-Learning")
        
        return env
    
    def evaluate_model_wandb(self, agent: QLearningAgent, eval_env, n_episodes, run_type):
        """Evaluates a given Q-Learning agent and logs results to WandB."""
        print(f"\n📊 Starting {run_type} evaluation...")
        episode_rewards = []
        success_count = 0
        collision_count = 0
        episode_lengths = []

        for episode in tqdm(range(n_episodes), desc=f"Evaluating Q-Learning ({run_type})"):
            obs, info = eval_env.reset()
            episode_reward = 0
            steps = 0
            done = truncated = False
            
            while not (done or truncated):
                action = agent.get_action(obs, training=False)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            if info.get('crashed', False):
                collision_count += 1
            elif info.get('arrived', False):
                success_count += 1

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = success_count / n_episodes
        collision_rate = collision_count / n_episodes
        mean_episode_length = np.mean(episode_lengths)

        metrics = {
            f"{run_type}/mean_reward": mean_reward,
            f"{run_type}/std_reward": std_reward,
            f"{run_type}/success_rate": success_rate,
            f"{run_type}/collision_rate": collision_rate,
            f"{run_type}/n_eval_episodes": n_episodes,
            f"{run_type}/mean_episode_length": mean_episode_length,
        }
        wandb.log(metrics)

        print(f"   Evaluation Results ({run_type}):")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Collision Rate: {collision_rate:.1%}")
        
        return metrics

    def train_q_learning_agent(self, total_episodes=5000, eval_freq=500, 
                              n_eval_episodes=20, show_training=True, 
                              stopping_mode="safety", max_steps_per_episode=300,
                              seed=0, load_model_path=None):
        
        if load_model_path:
            print(f"\n🔄 Loading Q-Learning model from {load_model_path} for evaluation...")
            try:
                agent = QLearningAgent.load_agent(load_model_path)
                print(f"✅ Model loaded successfully from {load_model_path}")
                
                eval_env = self.create_environment() # Create a fresh environment for evaluation
                metrics = self.evaluate_model_wandb(agent, eval_env, n_eval_episodes=n_eval_episodes, run_type="loaded_model_evaluation")
                eval_env.close()
                return [{"seed": seed, "algorithm": "Q_Learning", "evaluation_metrics": metrics, "run_type": "loaded_model_evaluation"}]
                
            except Exception as e:
                print(f"❌ Error loading model from {load_model_path}: {e}")
                return []

        print(f"🎯 Training Q-Learning Agent")
        print(f"📊 Total episodes: {total_episodes:,}")
        print(f"🔍 Evaluation frequency: every {eval_freq} episodes")
        print(f"🎬 Visual training: {'Enabled' if show_training else 'Disabled'}")
        print(f"🛡️ Stopping mode: {stopping_mode}")
        print("-" * 60)
        
        # Create environments
        if show_training:
            train_env = self.create_environment(render_mode="human")
        else:
            train_env = self.create_environment()
        
        eval_env = self.create_environment()
        
        # Set seeds
        train_env.reset(seed=seed)
        eval_env.reset(seed=seed + 1000)
        np.random.seed(seed)
        
        # Create Q-Learning agent
        agent = create_q_learning_agent(
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.995
        )
        
        # Training tracking
        training_rewards = []
        evaluation_history = []
        best_success_rate = 0
        episodes_since_improvement = 0
        start_time = time.time()
        
        # Safety stopping criteria (same as PPO)
        min_success_rate = 0.95 if stopping_mode == "safety" else 0.8
        max_collision_rate = 0.05 if stopping_mode == "safety" else 0.2
        min_evaluations_before_stop = 5
        
        print(f"🛡️ Safety criteria: {min_success_rate:.1%} success, <{max_collision_rate:.1%} collision")
        
        # Training loop
        pbar = tqdm(total=total_episodes, desc="Training Q-Learning", unit="episode")
        
        for episode in range(total_episodes):
            # Training episode
            obs, info = train_env.reset()
            episode_reward = 0
            steps = 0
            done = truncated = False
            
            while not (done or truncated) and steps < max_steps_per_episode:
                # Get action
                action = agent.get_action(obs, training=True)
                next_obs, reward, done, truncated, info = train_env.step(action)
                
                # Update Q-table
                agent.update_q_table(obs, action, reward, next_obs, done or truncated)
                
                obs = next_obs
                episode_reward += reward
                steps += 1
            
            # Update agent stats
            success = info.get('arrived', False)
            collision = info.get('crashed', False)
            agent.update_episode_stats(episode_reward, success, collision)
            agent.decay_epsilon()
            
            training_rewards.append(episode_reward)
            
            # Progress tracking
            pbar.set_postfix({
                'Reward': f"{episode_reward:.1f}",
                'Epsilon': f"{agent.epsilon:.3f}",
                'Q-Size': len(agent.q_table)
            })
            pbar.update(1)
            
            # Evaluation
            if episode % eval_freq == 0 or episode == total_episodes - 1:
                print(f"\n📊 Evaluation at episode {episode}")
                
                eval_metrics = self.evaluate_q_agent(agent, eval_env, n_eval_episodes)
                eval_metrics['episode'] = episode
                eval_metrics['training_time'] = time.time() - start_time
                eval_metrics['q_table_size'] = len(agent.q_table)
                eval_metrics['exploration_rate'] = agent.epsilon
                
                evaluation_history.append(eval_metrics)
                wandb.log(eval_metrics)

                # Print evaluation results
                print(f"   Mean Reward: {eval_metrics['mean_reward']:.2f}")
                print(f"   Success Rate: {eval_metrics['success_rate']:.1%}")
                print(f"   Collision Rate: {eval_metrics['collision_rate']:.1%}")
                print(f"   Q-table Size: {eval_metrics['q_table_size']} states")
                
                # Check for improvement
                if eval_metrics['success_rate'] > best_success_rate:
                    best_success_rate = eval_metrics['success_rate']
                    episodes_since_improvement = 0
                    
                    # Save best model
                    model_path = f"experiments/results/models/Q_Learning_best_seed_{seed}.pkl"
                    agent.save_agent(model_path)
                    artifact = wandb.Artifact(name=f"Q_Learning_best_seed_{seed}", type="model")
                    artifact.add_file(model_path)
                    wandb.log_artifact(artifact)
                    print(f"✅ Best Q-Learning model saved to WandB as artifact: {artifact.name}")
                else:
                    episodes_since_improvement += eval_freq
                
                # Check stopping criteria
                if stopping_mode == "safety" and len(evaluation_history) >= min_evaluations_before_stop:
                    recent_success = eval_metrics['success_rate']
                    recent_collision = eval_metrics['collision_rate']
                    
                    if (recent_success >= min_success_rate and 
                        recent_collision <= max_collision_rate):
                        
                        print(f"\n🎉 SAFETY TARGET ACHIEVED!")
                        print(f"   ✅ Success Rate: {recent_success:.1%} >= {min_success_rate:.1%}")
                        print(f"   ✅ Collision Rate: {recent_collision:.1%} <= {max_collision_rate:.1%}")
                        print(f"   🛡️ Training stopped for robust, safe Q-Learning model!")
                        break
                
                # Early stopping if no improvement
                if episodes_since_improvement > eval_freq * 10:  # 10 evaluations without improvement
                    print(f"\n⏹️ Early stopping: No improvement for {episodes_since_improvement} episodes")
                    break
        
        pbar.close()
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🏁 Final Evaluation")
        final_metrics = self.evaluate_q_agent(agent, eval_env, n_episodes=50, render=False)
        
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📊 Final Performance:")
        print(f"   Success Rate: {final_metrics['success_rate']:.1%}")
        print(f"   Collision Rate: {final_metrics['collision_rate']:.1%}")
        print(f"   Mean Reward: {final_metrics['mean_reward']:.2f}")
        print(f"   Q-table Size: {len(agent.q_table)} states")
        
        # Save final model
        final_model_path = f"experiments/results/models/Q_Learning_final_seed_{seed}.pkl"
        agent.save_agent(final_model_path)
        artifact = wandb.Artifact(name=f"Q_Learning_final_seed_{seed}", type="model")
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)
        print(f"✅ Final Q-Learning model saved to WandB as artifact: {artifact.name}")
        
        # Save training results
        results = {
            'algorithm': 'Q_Learning',
            'seed': seed,
            'total_episodes': episode + 1,
            'training_time': training_time,
            'training_rewards': training_rewards,
            'evaluation_history': evaluation_history,
            'final_metrics': final_metrics,
            'q_table_summary': agent.get_q_table_summary(),
            'hyperparameters': {
                'learning_rate': agent.learning_rate,
                'discount_factor': agent.discount_factor,
                'epsilon_decay': agent.epsilon_decay,
                'final_epsilon': agent.epsilon
            },
            'environment': {
                'name': self.env_name,
                'custom_rewards': self.use_custom_env
            }
        }
        
        # Save results to JSON
        results_path = f"experiments/results/Q_Learning_training_results_seed_{seed}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_path}")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return results


def main():
    global TOTAL_EPISODES, EVAL_FREQUENCY, N_EVAL_EPISODES, MAX_EPISODE_STEPS, SEED, STOPPING_MODE, USE_CUSTOM_REWARDS, LOAD_MODEL_PATH
    
    print("🎯 Q-LEARNING TRAINING FOR AUTONOMOUS VEHICLES")
    print("=" * 60)
    
    # Initialize Weights & Biases in offline mode
    wandb.init(
        project="rl4avs_testing", # You can change this project name
        group="Q_Learning",
        mode="offline",  # Run in offline mode to avoid authentication issues
        config={
            "total_episodes": TOTAL_EPISODES,
            "eval_frequency": EVAL_FREQUENCY,
            "n_eval_episodes": N_EVAL_EPISODES,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "seed": SEED,
            "stopping_mode": STOPPING_MODE,
            "use_custom_rewards": USE_CUSTOM_REWARDS,
            "load_model_path": LOAD_MODEL_PATH,
            "agent_hyperparameters": {
                "learning_rate": 0.1,
                "discount_factor": 0.99,
                "epsilon_decay": 0.995,
            }
        }
    )

    print("🔬 This creates a Q-Learning baseline for comparison with PPO")
    print("🎚️ Uses the same environment, actions, and evaluation metrics")
    print("=" * 60)
    
    # ===============================
    # 🎯 Q-LEARNING CONFIGURATION
    # ===============================
    
    # Training Configuration
    TOTAL_EPISODES = 5000      # Equivalent to ~250K steps (50 steps/episode avg)
    EVAL_FREQUENCY = 500       # Evaluate every 500 episodes
    N_EVAL_EPISODES = 20       # Episodes per evaluation
    MAX_EPISODE_STEPS = 300    # Maximum steps per episode
    SEED = 0
    
    # Display Configuration
    SHOW_TRAINING = True       # Show training with rendering
    
    # Stopping Configuration
    STOPPING_MODE = "safety"   # Same as PPO: "safety", "episodes", "reward"
    
    # Environment Configuration
    USE_CUSTOM_REWARDS = True  # Same custom environment as PPO
    
    print("🎯 Q-LEARNING TRAINING CONFIGURATION:")
    print(f"   Total episodes: {TOTAL_EPISODES:,}")
    print(f"   Evaluation frequency: every {EVAL_FREQUENCY} episodes")
    print(f"   Max steps per episode: {MAX_EPISODE_STEPS}")
    print(f"   Stopping mode: {STOPPING_MODE}")
    print(f"   Custom rewards: {'ENABLED' if USE_CUSTOM_REWARDS else 'DISABLED'}")
    
    if USE_CUSTOM_REWARDS:
        print("\n🚀 ENHANCED REWARD FEATURES (same as PPO):")
        print("   • Strong idle penalty (-0.3 per step)")
        print("   • Stationary penalty (-1.0+ exponential)")
        print("   • High completion reward (+15.0)")
        print("   • Progress rewards for forward movement")
        print("   • Efficiency bonus for fast completion")
        print("   • Collision detection and penalties")
    
    print("\n🎚️ Q-LEARNING HYPERPARAMETERS:")
    print("   • Learning rate: 0.1")
    print("   • Discount factor: 0.99")
    print("   • Initial epsilon: 1.0")
    print("   • Epsilon decay: 0.995")
    print("   • State discretization: 5-6 dimensional")
    
    print("\n📊 EXPECTED Q-TABLE SIZE:")
    print("   • Vehicle count bins: 3")
    print("   • Distance bins: 5")
    print("   • Speed bins: 4")
    print("   • Angle bins: 8")
    print("   • Position bins: 4")
    print("   • Estimated max states: ~1,920")
    print("   • Actual visited states: typically 200-800")
    
    # Initialize experiment runner
    runner = QLearningExperimentRunner(
        env_name="roundabout-v0",
        use_custom_env=USE_CUSTOM_REWARDS
    )
    
    input("🎬 Press Enter to start Q-Learning training (ensure you can see pygame windows)...")
    
    # Train Q-Learning agent
    results = runner.train_q_learning_agent(
        total_episodes=TOTAL_EPISODES,
        eval_freq=EVAL_FREQUENCY,
        n_eval_episodes=N_EVAL_EPISODES,
        show_training=SHOW_TRAINING,
        stopping_mode=STOPPING_MODE,
        max_steps_per_episode=MAX_EPISODE_STEPS,
        seed=SEED
    )
    
    print(f"\n🎉 Q-Learning training completed!")
    print("📊 Results summary:")
    print(f"   • Final success rate: {results['final_metrics']['success_rate']:.1%}")
    print(f"   • Final collision rate: {results['final_metrics']['collision_rate']:.1%}")
    print(f"   • Final mean reward: {results['final_metrics']['mean_reward']:.2f}")
    print(f"   • Q-table size: {results['q_table_summary']['size']} states")
    print(f"   • Training time: {results['training_time']:.1f} seconds")
    
    print("\n💾 Files created:")
    print("   • experiments/results/models/Q_Learning_final_seed_0.pkl")
    print("   • experiments/results/models/Q_Learning_best_seed_0.pkl")
    print("   • experiments/results/Q_Learning_training_results_seed_0.json")
    
    print("\n💡 Next steps:")
    print("   • Test Q-Learning model: python experiments/watch_q_learning.py")
    print("   • Compare with PPO: python experiments/compare_algorithms.py")
    print("   • Analyze Q-table: python scratch/analyze_q_table.py")

    if LOAD_MODEL_PATH:
        print("📊 Model evaluation completed!")
        print("   • Check WandB for 'loaded_model_evaluation/' metrics.")
    else:
        print(f"\n🏆 Q-Learning baseline established for comparison with PPO!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning agent with various stopping criteria.")
    parser.add_argument("--total_episodes", type=int, default=5000,
                        help="Total episodes for training.")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds to run.")
    parser.add_argument("--stopping_mode", type=str, default="safety",
                        choices=["safety", "episodes", "reward"],
                        help="Stopping criteria for training.")
    parser.add_argument("--max_steps_per_episode", type=int, default=300,
                        help="Maximum steps per episode.")
    parser.add_argument("--use_custom_rewards", action="store_true",
                        help="Use the custom environment with enhanced reward system.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Path to load a pre-trained Q-Learning model for evaluation.")
    
    args = parser.parse_args()

    # Make variables global so they can be accessed in main()
    global TOTAL_EPISODES, EVAL_FREQUENCY, N_EVAL_EPISODES, MAX_EPISODE_STEPS, SEED, STOPPING_MODE, USE_CUSTOM_REWARDS, LOAD_MODEL_PATH
    TOTAL_EPISODES = args.total_episodes
    N_SEEDS = args.n_seeds
    STOPPING_MODE = args.stopping_mode
    MAX_EPISODE_STEPS = args.max_steps_per_episode
    USE_CUSTOM_REWARDS = args.use_custom_rewards
    SEED = args.seed
    LOAD_MODEL_PATH = args.load_model_path
    
    # Set default values for other variables
    EVAL_FREQUENCY = 100  # Default evaluation frequency
    N_EVAL_EPISODES = 5   # Default number of evaluation episodes

    main()
    wandb.finish() # End WandB run
