import subprocess
import os
import sys
import argparse

# Add the parent directory to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Define the paths to the test training scripts
PPO_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_ppo_test.py')
A2C_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_a2c_test.py')
DQN_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_dqn_test.py')
Q_LEARNING_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_q_learning_test.py')

def run_experiment(script_path, total_timesteps, n_seeds, stopping_mode, extended_training, use_custom_rewards, algorithm, load_model_path=None):
    """Runs a single experiment with the given parameters."""
    print(f"\n🚀 Starting {algorithm} experiment with {{'total_timesteps': {total_timesteps}, 'n_seeds': {n_seeds}, 'stopping_mode': '{stopping_mode}', 'extended_training': {extended_training}, 'use_custom_rewards': {use_custom_rewards}, 'load_model_path': '{load_model_path}'}}")
    
    cmd = [
        "python3", script_path,
        "--total_timesteps", str(total_timesteps),
        "--n_seeds", str(n_seeds),
        "--stopping_mode", stopping_mode,
    ]

    if extended_training:
        cmd.append("--extended_training")
    if use_custom_rewards:
        cmd.append("--use_custom_rewards")
    if load_model_path:
        cmd.extend(["--load_model_path", load_model_path])

    try:
        # Run the script as a subprocess
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {algorithm} experiment completed successfully.")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ {algorithm} experiment failed.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
    except FileNotFoundError:
        print(f"❌ Error: Python interpreter or script not found. Make sure python3 is in your PATH and scripts exist.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run RL training experiments with custom settings and WandB integration.")
    parser.add_argument("--algorithm", type=str, default="all", 
                        choices=["ppo", "a2c", "dqn", "q_learning", "all"],
                        help="Specify which algorithm to test (ppo, a2c, dqn, q_learning, or all).")
    parser.add_argument("--total_timesteps", type=int, default=10000,
                        help="Total timesteps for training each algorithm.")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds to run for each algorithm.")
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

    # Modify the training scripts to accept arguments
    # I will add this logic in the next steps as part of modifying each script

    if args.algorithm == "all":
        # Run all algorithms
        print("\n--- Running ALL experiments ---")
        run_experiment(PPO_SCRIPT, args.total_timesteps, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, "PPO", args.load_model_path)
        run_experiment(A2C_SCRIPT, args.total_timesteps, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, "A2C", args.load_model_path)
        run_experiment(DQN_SCRIPT, args.total_timesteps, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, "DQN", args.load_model_path)
        # Q-Learning uses total_episodes, not total_timesteps, need to adjust this
        run_experiment(Q_LEARNING_SCRIPT, args.total_timesteps // 50, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, "Q_Learning", args.load_model_path) # Assuming 50 steps/episode
    else:
        # Run a specific algorithm
        script_map = {
            "ppo": PPO_SCRIPT,
            "a2c": A2C_SCRIPT,
            "dqn": DQN_SCRIPT,
            "q_learning": Q_LEARNING_SCRIPT,
        }
        script_to_run = script_map.get(args.algorithm)
        if script_to_run:
            if args.algorithm == "q_learning":
                run_experiment(script_to_run, args.total_timesteps // 50, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, "Q_Learning", args.load_model_path)
            else:
                run_experiment(script_to_run, args.total_timesteps, args.n_seeds, args.stopping_mode, args.extended_training, args.use_custom_rewards, args.algorithm.upper(), args.load_model_path)
        else:
            print(f"Error: Invalid algorithm specified: {args.algorithm}")

if __name__ == "__main__":
    main()
