#!/usr/bin/env python3
"""
🛡️ ROBUST RL TRAINING FOR AUTONOMOUS VEHICLES 🛡️

This enhanced training script provides multiple stopping criteria focused on
safety and route completion rather than just reward maximization.

🎯 STOPPING MODES AVAILABLE:

1. "safety" (RECOMMENDED) 
   - Stops when: 95% success rate AND <5% collision rate
   - Best for: Reliable, everyday driving scenarios
   - Use case: When you need consistent route completion

2. "progressive"
   - Starts lenient (70% success), becomes stricter (95% success)
   - Best for: Understanding learning progression
   - Use case: Research on learning curves and adaptation

3. "ultra_safe" 
   - Stops when: 98% success rate AND <2% collision rate
   - Best for: Critical safety applications
   - Use case: When failure is not an option

4. "reward" 
   - Original reward threshold stopping (reward >= 2.5)
   - Best for: Comparison with traditional methods
   - Use case: Academic comparisons, less reliable for safety

5. "extended"
   - No early stopping, trains for full timesteps
   - Best for: Maximum performance extraction
   - Use case: When you want to see absolute limits

🚀 QUICK START:
   1. Change STOPPING_MODE in main() to your preferred mode
   2. Adjust TOTAL_TIMESTEPS (50K recommended for robust models)
   3. Set EXTENDED_TRAINING = True to disable early stopping
   4. Run: python experiments/train_ppo.py

📊 MONITORING:
   - Real-time success/collision rates during training
   - Automatic model saving when criteria are met
   - Detailed action analysis and debugging

🎓 FOR MSC RESEARCH:
   - Use "safety" for main results
   - Use "progressive" to show learning curves  
   - Use "ultra_safe" for safety-critical analysis
   - Use "extended" for ablation studies
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the project root to sys.path if not already there, for resolving 'src' imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents import ExperimentRunner
from src.visualization_callback import VisualizationCallbackWithDebug
from src.environment import make_env
from src.utils import print_experiment_summary
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from tqdm import tqdm
import time
import wandb # Import wandb
import argparse # Import argparse
from stable_baselines3 import PPO # Import PPO for loading


class SafetyBasedStoppingCallback(BaseCallback):
    """
    Custom callback that stops training based on safety and success metrics
    instead of just reward threshold
    """
    
    def __init__(self, eval_env, check_freq=2000, n_eval_episodes=20, 
                 min_success_rate=0.95, max_collision_rate=0.05, 
                 min_episodes_before_stop=50, verbose=1):
        super(SafetyBasedStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.min_success_rate = min_success_rate
        self.max_collision_rate = max_collision_rate
        self.min_episodes_before_stop = min_episodes_before_stop
        self.episode_count = 0
        
        # Tracking metrics
        self.success_rates = []
        self.collision_rates = []
        self.mean_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.episode_count += 1
            
            # Don't stop too early
            if self.episode_count < self.min_episodes_before_stop // (self.check_freq // 1000):
                return True
            
            # Evaluate current policy
            success_count = 0
            collision_count = 0
            episode_rewards = []
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0
                done = truncated = False
                
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                
                # Check episode outcome
                if info.get('crashed', False):
                    collision_count += 1
                elif info.get('arrived', False):
                    success_count += 1
            
            # Calculate metrics
            success_rate = success_count / self.n_eval_episodes
            collision_rate = collision_count / self.n_eval_episodes
            mean_reward = np.mean(episode_rewards)
            
            # Store metrics
            self.success_rates.append(success_rate)
            self.collision_rates.append(collision_rate)
            self.mean_rewards.append(mean_reward)

            # Log metrics to WandB
            wandb.log({
                "safety_evaluation/success_rate": success_rate,
                "safety_evaluation/collision_rate": collision_rate,
                "safety_evaluation/mean_reward": mean_reward,
                "safety_evaluation/episode_count": self.episode_count,
                "global_step": self.n_calls
            })

            if self.verbose > 0:
                print(f"\n📊 Safety Evaluation (Step {self.n_calls}):")
                print(f"   Success Rate: {success_rate:.1%} (target: {self.min_success_rate:.1%})")
                print(f"   Collision Rate: {collision_rate:.1%} (max: {self.max_collision_rate:.1%})")
                print(f"   Mean Reward: {mean_reward:.3f}")
            
            # Check stopping criteria
            if (success_rate >= self.min_success_rate and 
                collision_rate <= self.max_collision_rate):
                
                print(f"\n🎉 SAFETY TARGET ACHIEVED!")
                print(f"   ✅ Success Rate: {success_rate:.1%} >= {self.min_success_rate:.1%}")
                print(f"   ✅ Collision Rate: {collision_rate:.1%} <= {self.max_collision_rate:.1%}")
                print(f"   🛡️ Training stopped for robust, safe model!")
                return False  # Stop training
        
        return True  # Continue training


class ProgressiveRobustnessCallback(BaseCallback):
    """
    Callback that uses progressive criteria - starts lenient, becomes stricter
    """
    
    def __init__(self, eval_env, check_freq=2000, n_eval_episodes=15, verbose=1):
        super(ProgressiveRobustnessCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluation_count = 0
        
    def _get_current_criteria(self):
        """Progressive criteria that get stricter over time"""
        if self.evaluation_count < 3:
            return 0.7, 0.3  # 70% success, max 30% collision
        elif self.evaluation_count < 6:
            return 0.85, 0.15  # 85% success, max 15% collision  
        else:
            return 0.95, 0.05  # 95% success, max 5% collision
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.evaluation_count += 1
            min_success_rate, max_collision_rate = self._get_current_criteria()
            
            # Evaluate current policy
            success_count = 0
            collision_count = 0
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = truncated = False
                
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                
                if info.get('crashed', False):
                    collision_count += 1
                elif info.get('arrived', False):
                    success_count += 1
            
            success_rate = success_count / self.n_eval_episodes
            collision_rate = collision_count / self.n_eval_episodes

            # Log metrics to WandB
            wandb.log({
                "progressive_evaluation/success_rate": success_rate,
                "progressive_evaluation/collision_rate": collision_rate,
                "progressive_evaluation/min_success_criteria": min_success_rate,
                "progressive_evaluation/max_collision_criteria": max_collision_rate,
                "progressive_evaluation/evaluation_count": self.evaluation_count,
                "global_step": self.n_calls
            })

            if self.verbose > 0:
                print(f"\n📈 Progressive Evaluation #{self.evaluation_count}:")
                print(f"   Current Criteria: {min_success_rate:.1%} success, max {max_collision_rate:.1%} collision")
                print(f"   Achieved: {success_rate:.1%} success, {collision_rate:.1%} collision")
            
            # Only check final criteria after several evaluations
            if (self.evaluation_count >= 6 and 
                success_rate >= min_success_rate and 
                collision_rate <= max_collision_rate):
                
                print(f"\n🎯 PROGRESSIVE TARGET ACHIEVED!")
                print(f"   Final Success Rate: {success_rate:.1%}")
                print(f"   Final Collision Rate: {collision_rate:.1%}")
                return False
        
        return True
import numpy as np


class DebugExperimentRunner(ExperimentRunner):
    """Enhanced experiment runner with progress bars and action debugging"""
    
    def __init__(self, env_name="roundabout-v0", use_custom_env=False):
        super().__init__(env_name, use_custom_env)
        # Ensure debug-specific directories exist
        os.makedirs("experiments/results/models", exist_ok=True)
        os.makedirs("experiments/results/logs", exist_ok=True)
    
    def create_environment(self, custom=None):
        """Create environment using the make_env function"""
        from src.environment import make_env
        if custom is None:
            custom = self.use_custom_env
        return make_env(self.env_name, custom=custom)

    def evaluate_model_wandb(self, model, eval_env, n_eval_episodes, algo_name, total_timesteps, run_type):
        """Evaluates a given model and logs results to WandB."""
        print(f"\n📊 Starting {run_type} evaluation for {algo_name}...")
        success_count = 0
        collision_count = 0
        episode_rewards = []
        
        for _ in tqdm(range(n_eval_episodes), desc=f"Evaluating {algo_name} ({run_type})"):
            obs, info = eval_env.reset()
            episode_reward = 0
            done = truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            if info.get('crashed', False):
                collision_count += 1
            elif info.get('arrived', False):
                success_count += 1
        
        success_rate = success_count / n_eval_episodes
        collision_rate = collision_count / n_eval_episodes
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        metrics = {
            f"{run_type}/mean_reward": mean_reward,
            f"{run_type}/std_reward": std_reward,
            f"{run_type}/success_rate": success_rate,
            f"{run_type}/collision_rate": collision_rate,
            f"{run_type}/n_eval_episodes": n_eval_episodes,
            f"{run_type}/total_timesteps_evaluated": total_timesteps # This reflects the number of environment steps in evaluation.
        }
        wandb.log(metrics)

        print(f"   Evaluation Results ({run_type}):")
        print(f"   Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Collision Rate: {collision_rate:.1%}")
        
        return metrics

    def train_algorithm_with_debug(self, algo_name, total_timesteps=100000,
                                   n_seeds=1, show_training=True, debug_actions=True,
                                   stopping_mode="safety", extended_training=False,
                                   policy_kwargs=None, load_model_path=None):
        
        if load_model_path:
            print(f"\n🔄 Loading {algo_name} model from {load_model_path} for evaluation...")
            try:
                # Need to create a dummy environment to load the model correctly
                # The environment definition is crucial for the model to be loaded correctly.
                # Assuming the environment is consistent with the one used for training the model.
                if self.use_custom_env:
                    from src.environment import register_custom_env
                    register_custom_env()
                    env_for_loading = gym.make('custom-roundabout-v0')
                else:
                    env_for_loading = gym.make(self.env_name)
                
                model = PPO.load(load_model_path, env=env_for_loading) # Specify algorithm type for loading
                print(f"✅ Model loaded successfully from {load_model_path}")
                
                # Perform evaluation only
                eval_env = self.create_environment() # Create a fresh environment for evaluation
                metrics = self.evaluate_model_wandb(model, eval_env, n_eval_episodes=100, algo_name=algo_name, total_timesteps=total_timesteps, run_type="loaded_model_evaluation")
                eval_env.close()
                return [{"seed": 0, "algorithm": algo_name, "evaluation_metrics": metrics, "run_type": "loaded_model_evaluation"}]
                
            except Exception as e:
                print(f"❌ Error loading model from {load_model_path}: {e}")
                return []

        # Original training logic follows if no model path is provided
        print(f"🎯 Training {algo_name} with comprehensive debugging")
        print(f"📊 Total timesteps: {total_timesteps:,}")
        print(f"🔍 Action debugging: {'Enabled' if debug_actions else 'Disabled'}")
        print(f"🎬 Visual training: {'Enabled' if show_training else 'Disabled'}")
        print(f"🛡️ Stopping mode: {stopping_mode}")
        print(f"⏱️ Extended training: {'Yes' if extended_training else 'No'}")
        print("-" * 60)

        seed_results = []

        # Overall progress for all seeds
        overall_pbar = tqdm(
            total=n_seeds,
            desc="Training Seeds",
            unit="seed",
            position=0
        )

        for seed in range(n_seeds):
            overall_pbar.set_description(f"🌱 Training Seed {seed + 1}/{n_seeds}")

            # Create training environment (FIXED: maintain custom env even with rendering)
            if show_training and self.use_custom_env:
                # Create custom environment with rendering
                from src.environment import register_custom_env
                register_custom_env()
                train_env = gym.make('custom-roundabout-v0', render_mode="human")
                print("🎁 Using CUSTOM environment with rendering for training")
            elif show_training:
                train_env = gym.make(self.env_name, render_mode="human")
                print("⚠️ Using STANDARD environment with rendering for training")
            else:
                train_env = make_env(self.env_name, custom=self.use_custom_env)
                env_type = "CUSTOM" if self.use_custom_env else "STANDARD"
                print(f"🎯 Using {env_type} environment for training")

            train_env.reset(seed=seed)

            # Create evaluation environment (FIXED: use same environment type as training)
            if self.use_custom_env:
                from src.environment import register_custom_env
                register_custom_env()
                eval_env = gym.make('custom-roundabout-v0', render_mode="human")
                print("🎁 Using CUSTOM environment for evaluation")
            else:
                eval_env = gym.make(self.env_name, render_mode="human")
                print("⚠️ Using STANDARD environment for evaluation")
            eval_env.reset(seed=seed + 1000)

            # Set up logging
            log_dir = f"results/logs/{algo_name}_debug_seed_{seed}/"

            # Initialize model with tuned hyperparameters
            if algo_name == 'PPO' and policy_kwargs is not None:
                # Use advanced PPO hyperparameters
                model = self.algorithms[algo_name](
                    'MlpPolicy',
                    train_env,
                    verbose=0,
                    seed=seed,
                    tensorboard_log=log_dir,
                    **policy_kwargs  # Unpack all hyperparameters
                )
                print("🎛️ Using ADVANCED PPO hyperparameters for collision avoidance")
            else:
                # Use default hyperparameters
                model = self.algorithms[algo_name](
                    'MlpPolicy',
                    train_env,
                    verbose=0,
                    seed=seed,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs
                )
                print("⚙️ Using default hyperparameters")

            # Create enhanced debugging callback
            debug_callback = VisualizationCallbackWithDebug(
                eval_env=eval_env,
                eval_freq=20000,
                render_freq=10000,
                total_timesteps=total_timesteps,
                debug_actions=debug_actions,
                verbose=1
            )

            # Train with comprehensive debugging
            print(f"\n🚀 Starting training for seed {seed}")
            start_time = time.time()

            # Create evaluation environment for callbacks (FIXED: use same environment type)
            if self.use_custom_env:
                from src.environment import register_custom_env
                register_custom_env()
                eval_env = gym.make('custom-roundabout-v0')
                print("🎁 Using CUSTOM environment for safety evaluation")
            else:
                eval_env = gym.make(self.env_name)
                print("⚠️ Using STANDARD environment for safety evaluation")

            # Configure stopping criteria based on mode
            callbacks = [debug_callback]
            
            if not extended_training:  # Only add stopping callbacks if not in extended mode
                if stopping_mode == "safety":
                    # Safety-focused stopping: High success rate, low collision rate
                    safety_callback = SafetyBasedStoppingCallback(
                        eval_env=eval_env,
                        check_freq=3000,
                        n_eval_episodes=20,
                        min_success_rate=0.95,  # 95% success rate
                        max_collision_rate=0.05,  # Max 5% collision rate
                        min_episodes_before_stop=30,
                        verbose=1
                    )
                    callbacks.append(safety_callback)
                    print("🛡️ Using SAFETY-BASED stopping (95% success, <5% collision)")
                    
                elif stopping_mode == "progressive":
                    # Progressive criteria that get stricter over time
                    progressive_callback = ProgressiveRobustnessCallback(
                        eval_env=eval_env,
                        check_freq=2500,
                        n_eval_episodes=15,
                        verbose=1
                    )
                    callbacks.append(progressive_callback)
                    print("📈 Using PROGRESSIVE stopping (gradually stricter criteria)")
                    
                elif stopping_mode == "ultra_safe":
                    # Ultra-conservative: Near perfect performance required
                    ultra_safe_callback = SafetyBasedStoppingCallback(
                        eval_env=eval_env,
                        check_freq=2000,
                        n_eval_episodes=25,
                        min_success_rate=0.98,  # 98% success rate
                        max_collision_rate=0.02,  # Max 2% collision rate
                        min_episodes_before_stop=50,
                        verbose=1
                    )
                    callbacks.append(ultra_safe_callback)
                    print("🚨 Using ULTRA-SAFE stopping (98% success, <2% collision)")
                    
                elif stopping_mode == "reward":
                    # Original reward-based stopping (less reliable for safety)
                    stop_callback = StopTrainingOnRewardThreshold(
                        reward_threshold=2.5,  # Increased threshold
                        verbose=1
                    )
                    eval_callback = EvalCallback(
                        eval_env,
                        eval_freq=1000,
                        callback_on_new_best=stop_callback,
                        verbose=1
                    )
                    callbacks.append(eval_callback)
                    print("⚠️ Using REWARD-BASED stopping (less robust)")
            else:
                print("⏱️ EXTENDED TRAINING mode: No early stopping, training full duration")

            # Combine all callbacks
            combined_callbacks = CallbackList(callbacks)

            # Train the model
            actual_timesteps = total_timesteps if extended_training else min(total_timesteps, 50000)
            print(f"🚀 Training for {actual_timesteps:,} timesteps...")
            
            model.learn(
                total_timesteps=actual_timesteps,
                callback=combined_callbacks
            )

            training_time = time.time() - start_time

            print(f"✅ Seed {seed} completed in {training_time:.1f} seconds")

            # Save model with proper path handling
            model_save_path = f"experiments/results/models/{algo_name}_debug_seed_{seed}"
            try:
                model.save(model_save_path)
                print(f"✅ Model saved to {model_save_path}")
                # Save model as a WandB artifact
                artifact = wandb.Artifact(name=f"{algo_name}_seed_{seed}", type="model")
                artifact.add_file(f"{model_save_path}.zip") # Stable Baselines3 saves as .zip
                wandb.log_artifact(artifact)
                print(f"✅ Model also saved to WandB as artifact: {artifact.name}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
                # Try alternative path
                os.makedirs("results/models", exist_ok=True)
                fallback_path = f"results/models/{algo_name}_debug_seed_{seed}"
                model.save(fallback_path)
                print(f"✅ Model saved to fallback path: {fallback_path}")

            # Clean up environments
            train_env.close()
            eval_env.close()

            result = {
                'seed': seed,
                'algorithm': algo_name,
                'total_timesteps': total_timesteps,
                'training_time': training_time,
                'debug_enabled': debug_actions
            }

            seed_results.append(result)
            overall_pbar.update(1)

        overall_pbar.close()
        self.results[algo_name] = seed_results
        return seed_results


def main():
    global TOTAL_TIMESTEPS, N_SEEDS, STOPPING_MODE, EXTENDED_TRAINING, USE_CUSTOM_REWARDS, LOAD_MODEL_PATH
    
    print("🎮 Enhanced RL Training with Action Debugging")
    print("=" * 60)
    
    # Configuration
    ALGORITHM = "PPO"
    
    # Initialize Weights & Biases in offline mode
    wandb.init(
        project="rl4avs_testing", # You can change this project name
        group=ALGORITHM,
        mode="offline",  # Run in offline mode to avoid authentication issues
        config={
            "algorithm": ALGORITHM,
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_seeds": N_SEEDS,
            "stopping_mode": STOPPING_MODE,
            "extended_training": EXTENDED_TRAINING,
            "use_custom_rewards": USE_CUSTOM_REWARDS,
            "load_model_path": LOAD_MODEL_PATH,
            "ppo_policy_kwargs": PPO_POLICY_KWARGS if ALGORITHM == 'PPO' else None,
            "ppo_advanced_kwargs": PPO_ADVANCED_KWARGS if ALGORITHM == 'PPO' else None,
        }
    )

    print("Features enabled:")
    print("  ✅ Real-time progress bars")
    print("  ✅ Action distribution tracking")
    print("  ✅ Reward trend monitoring")
    print("  ✅ Visual debugging during evaluation")
    print("  ✅ Comprehensive action analysis")
    print("=" * 60)

    # ===============================
    # 🛡️ ROBUST TRAINING CONFIGURATION
    # ===============================
    
    # Basic Configuration
    TOTAL_TIMESTEPS = 250000  # 5x more training for robustness
    N_SEEDS = 1
    ALGORITHM = 'PPO'  # PPO typically works best for continuous control
    SHOW_TRAINING = True
    DEBUG_ACTIONS = True
    
    # 🧠 HYPERPARAMETER TUNING: Deeper network for more complex decisions
    # Default is [64, 64]. We use a deeper network to learn more complex patterns.
    PPO_POLICY_KWARGS = dict(net_arch=[256, 256])
    
    # 🎛️ ADVANCED PPO HYPERPARAMETERS FOR COLLISION AVOIDANCE
    # These are specifically tuned for autonomous driving scenarios
    PPO_ADVANCED_KWARGS = dict(
        learning_rate=3e-4,              # Standard learning rate
        n_steps=2048,                    # Steps per update (increase for more stable gradients)
        batch_size=64,                   # Mini-batch size (smaller for more frequent updates)
        n_epochs=10,                     # Epochs per update (more training per batch)
        gamma=0.99,                      # Discount factor (prioritize future rewards)
        gae_lambda=0.95,                 # GAE parameter (bias-variance tradeoff)
        clip_range=0.2,                  # PPO clip parameter (prevent large policy changes)
        ent_coef=0.01,                   # Entropy coefficient (encourage exploration)
        vf_coef=0.5,                     # Value function coefficient
        max_grad_norm=0.5,               # Gradient clipping (prevent exploding gradients)
        target_kl=0.01,                  # Early stopping for KL divergence
        policy_kwargs=PPO_POLICY_KWARGS
    )
    
    # 🎯 STOPPING MODE OPTIONS:
    # "safety"     - Stop when 95% success rate + <5% collision rate (RECOMMENDED)
    # "progressive"- Start lenient, become stricter (good for learning curve)
    # "ultra_safe" - Stop when 98% success rate + <2% collision rate (ultra robust)
    # "reward"     - Original reward-based stopping (less reliable)
    # "extended"   - No early stopping, train full duration (for comparison)
    
    STOPPING_MODE = "safety"  # ← CHANGE THIS TO TEST DIFFERENT APPROACHES
    EXTENDED_TRAINING = False  # Set True for no early stopping
    
    # 🔧 REWARD TUNING: Use custom environment for better rewards
    USE_CUSTOM_REWARDS = True  # ← Set to True for enhanced reward system
    
    print("🛡️ ROBUST TRAINING MODES AVAILABLE:")
    print("   • safety     - 95% success, <5% collision (recommended)")
    print("   • progressive- Gradually stricter criteria")  
    print("   • ultra_safe - 98% success, <2% collision")
    print("   • reward     - Original reward threshold")
    print("   • extended   - No early stopping")
    print(f"\n🎯 Current mode: {STOPPING_MODE}")
    print(f"🎁 Enhanced rewards: {'ENABLED' if USE_CUSTOM_REWARDS else 'DISABLED'}")
    
    if USE_CUSTOM_REWARDS:
        print("\n🚀 ENHANCED REWARD FEATURES:")
        print("   • 10x stronger idle penalty (-0.5 per step)")
        print("   • Massive stationary penalty (-2.0+ exponential)")
        print("   • 3x higher completion reward (+10.0)")
        print("   • Progress rewards for forward movement")
        print("   • Efficiency bonus for fast completion")
        print("   • Repeated mistake prevention")

    # Initialize enhanced experiment runner with custom rewards
    runner = DebugExperimentRunner(
        env_name="roundabout-v0",
        use_custom_env=USE_CUSTOM_REWARDS  # Use enhanced reward system
    )

    input("🎬 Press Enter to start enhanced training (ensure you can see pygame windows)...")

    # Train with robust stopping criteria
    results = runner.train_algorithm_with_debug(
        algo_name=ALGORITHM,
        total_timesteps=TOTAL_TIMESTEPS,
        n_seeds=N_SEEDS,
        show_training=SHOW_TRAINING,
        debug_actions=DEBUG_ACTIONS,
        stopping_mode=STOPPING_MODE,
        extended_training=EXTENDED_TRAINING,
        policy_kwargs=PPO_ADVANCED_KWARGS if ALGORITHM == 'PPO' else None,
        load_model_path=args.load_model_path
    )

    print(f"\n🎉 Robust training completed!")
    if args.load_model_path:
        print("📊 Model evaluation completed!")
        print("   • Check WandB for 'loaded_model_evaluation/' metrics.")
    else:
        print("📊 Results summary:")
        print("   • Check detailed action analysis above")
        print("   • Models saved in experiments/results/models/")
        print("   • TensorBoard logs in experiments/results/logs/")
    print("\n💡 Next steps:")
    print("   • Test your model: python experiments/watch_agents.py")
    print("   • Validate robustness: python scratch/test_model_loading.py")
    print("   • Monitor actions: python experiments/action_monitor.py")
    
    if STOPPING_MODE != "extended":
        print(f"\n🛡️ Training used '{STOPPING_MODE}' stopping criteria")
        print("   Your model should be robust for route completion!")
    else:
        print(f"\n⏱️ Training used full {TOTAL_TIMESTEPS:,} timesteps without early stopping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent with various stopping criteria.")
    parser.add_argument("--total_timesteps", type=int, default=250000,
                        help="Total timesteps for training.")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds to run.")
    parser.add_argument("--stopping_mode", type=str, default="safety",
                        choices=["safety", "progressive", "ultra_safe", "reward", "extended"],
                        help="Stopping criteria for training.")
    parser.add_argument("--extended_training", action="store_true",
                        help="Disable early stopping and train for full duration.")
    parser.add_argument("--use_custom_rewards", action="store_true",
                        help="Use the custom environment with enhanced reward system.")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Path to a pre-trained model to load for evaluation instead of training.")
    
    args = parser.parse_args()

    # Make variables global so they can be accessed in main()
    global TOTAL_TIMESTEPS, N_SEEDS, STOPPING_MODE, EXTENDED_TRAINING, USE_CUSTOM_REWARDS, LOAD_MODEL_PATH
    TOTAL_TIMESTEPS = args.total_timesteps
    N_SEEDS = args.n_seeds
    STOPPING_MODE = args.stopping_mode
    EXTENDED_TRAINING = args.extended_training
    USE_CUSTOM_REWARDS = args.use_custom_rewards
    LOAD_MODEL_PATH = args.load_model_path

    main()
    wandb.finish() # End WandB run

