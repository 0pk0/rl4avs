import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import ExperimentRunner
from src.visualization_callback import VisualizationCallbackWithDebug
from src.environment import make_env
from src.utils import print_experiment_summary
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
import gymnasium as gym
from tqdm import tqdm
import time


class DebugExperimentRunner(ExperimentRunner):
    """Enhanced experiment runner with progress bars and action debugging"""

    def train_algorithm_with_debug(self, algo_name, total_timesteps=100000,
                                   n_seeds=1, show_training=True, debug_actions=True):
        print(f"🎯 Training {algo_name} with comprehensive debugging")
        print(f"📊 Total timesteps: {total_timesteps:,}")
        print(f"🔍 Action debugging: {'Enabled' if debug_actions else 'Disabled'}")
        print(f"🎬 Visual training: {'Enabled' if show_training else 'Disabled'}")
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

            # Create training environment
            train_env = make_env(self.env_name, custom=self.use_custom_env)
            if show_training:
                train_env = gym.make(self.env_name, render_mode="human")

            train_env.reset(seed=seed)

            # Create evaluation environment
            eval_env = gym.make(self.env_name, render_mode="human")
            eval_env.reset(seed=seed + 1000)

            # Set up logging
            log_dir = f"results/logs/{algo_name}_debug_seed_{seed}/"

            # Initialize model
            model = self.algorithms[algo_name](
                'MlpPolicy',
                train_env,
                verbose=0,  # Disable verbose to keep terminal clean
                seed=seed,
                tensorboard_log=log_dir
            )

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

            # Create evaluation environment
            eval_env = gym.make(self.env_name)

            # Stop training when average reward reaches threshold
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=2.0,  # Adjust based on your environment
                verbose=1
            )

            # Evaluation callback that triggers the stop callback
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=1000,
                callback_on_new_best=stop_callback,
                verbose=1
            )

            # Combine with your existing debug callback
            callbacks = CallbackList([eval_callback, debug_callback])

            # Train with early stopping
            model.learn(
                total_timesteps=10000,
                callback=callbacks  # ← Use combined callbacks
            )

            training_time = time.time() - start_time

            print(f"✅ Seed {seed} completed in {training_time:.1f} seconds")

            # Save model
            model.save(f"results/models/{algo_name}_debug_seed_{seed}")

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
    print("🎮 Enhanced RL Training with Action Debugging")
    print("=" * 60)
    print("Features enabled:")
    print("  ✅ Real-time progress bars")
    print("  ✅ Action distribution tracking")
    print("  ✅ Reward trend monitoring")
    print("  ✅ Visual debugging during evaluation")
    print("  ✅ Comprehensive action analysis")
    print("=" * 60)

    # Configuration
    TOTAL_TIMESTEPS = 10000
    N_SEEDS = 1
    ALGORITHM = 'PPO'  # PPO typically works best
    SHOW_TRAINING = True
    DEBUG_ACTIONS = True

    # Initialize enhanced experiment runner
    runner = DebugExperimentRunner(
        env_name="roundabout-v0",
        use_custom_env=False
    )

    input("🎬 Press Enter to start enhanced training (ensure you can see pygame windows)...")

    # Train with full debugging
    results = runner.train_algorithm_with_debug(
        algo_name=ALGORITHM,
        total_timesteps=TOTAL_TIMESTEPS,
        n_seeds=N_SEEDS,
        show_training=SHOW_TRAINING,
        debug_actions=DEBUG_ACTIONS
    )

    print(f"\n Enhanced training completed!")
    print(" Check the detailed action analysis above")
    print(" Models saved in results/models/")
    print("TensorBoard logs in results/logs/")


if __name__ == "__main__":
    main()

