import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from collections import defaultdict, deque
import time
import numpy as np


def monitor_agent_actions(model_path, algorithm, n_episodes=3):
    """Real-time action monitoring for trained agent"""

    # Action mapping
    action_names = {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER"
    }

    # Load model
    if algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)

    env = gym.make("roundabout-v0", render_mode="human")

    print(f"🔍 Real-time Action Monitor for {algorithm}")
    print("Action Legend:")
    for action_id, name in action_names.items():
        print(f"  {action_id}: {name}")
    print("-" * 50)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        action_counts = defaultdict(int)
        step_count = 0

        print(f"\n🎬 Episode {episode + 1}/{n_episodes}")
        print("Real-time actions:")

        done = truncated = False
        while not (done or truncated):
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            action_value = action.item() if hasattr(action, 'item') else action

            # Log action
            episode_actions.append(action_value)
            action_counts[action_value] += 1
            step_count += 1

            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Display real-time action info
            action_name = action_names.get(action_value, f"Unknown_{action_value}")
            print(f"  Step {step_count:3d}: {action_name:10} | Reward: {reward:6.3f} | Total: {episode_reward:7.2f}")

            # Render
            env.render()
            time.sleep(0.1)  # Slow down for observation

        # Episode summary
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"  Total steps: {step_count}")
        print(f"  Final reward: {episode_reward:.3f}")

        if info.get('crashed', False):
            print(f"  ❌ Ended in collision")
        elif info.get('arrived', False):
            print(f"  ✅ Successfully completed roundabout")
        else:
            print(f"  ⏱️ Episode timeout")

        # Action distribution for this episode
        print(f"  Action distribution:")
        for action_id, count in sorted(action_counts.items()):
            percentage = (count / step_count) * 100
            action_name = action_names.get(action_id, f"Unknown_{action_id}")
            print(f"    {action_name}: {count} times ({percentage:.1f}%)")

        time.sleep(2)  # Pause between episodes

    env.close()


if __name__ == "__main__":
    # Configuration
    ALGORITHM = 'PPO'
    MODEL_PATH = f'results/models/{ALGORITHM}_debug_seed_0'
    N_EPISODES = 3

    try:
        monitor_agent_actions(MODEL_PATH, ALGORITHM, N_EPISODES)
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train a model first using the debug training script")
