import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
import time


def watch_trained_agent(model_path, algorithm, n_episodes=5, slow_motion=True):
    """Watch a trained agent navigate the roundabout"""

    # Load the trained model
    if algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Create environment with rendering
    env = gym.make("roundabout-v0", render_mode="human")

    print(f"🎬 Watching {algorithm} agent for {n_episodes} episodes...")
    print("📺 Observe the agent's behavior in the pygame window!")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = truncated = False

        print(f"\n🎯 Episode {episode + 1}/{n_episodes}")

        while not (done or truncated):
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Render the environment
            env.render()

            episode_reward += reward
            episode_length += 1

            # Add delay for better observation
            if slow_motion:
                time.sleep(0.05)

        # Episode summary
        print(f"  Length: {episode_length} steps, Reward: {episode_reward:.2f}")
        if info.get('crashed', False):
            print(f"  ❌ Episode ended in collision!")
        elif info.get('arrived', False):
            print(f"  ✅ Successfully navigated roundabout!")
        else:
            print(f"  ⏱️ Episode timed out")

        time.sleep(2)  # Pause between episodes

    env.close()
    print("🎬 Finished watching agent!")


def main():
    print("🎭 Agent Viewer")
    print("=" * 30)

    # Configuration
    ALGORITHM = 'DQN'  # Change this to match your trained model
    MODEL_PATH = f'results/models/{ALGORITHM}_seed_0'  # Adjust path as needed
    N_EPISODES = 3

    try:
        watch_trained_agent(
            model_path=MODEL_PATH,
            algorithm=ALGORITHM,
            n_episodes=N_EPISODES,
            slow_motion=True
        )
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train a model first using train_agents.py")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
