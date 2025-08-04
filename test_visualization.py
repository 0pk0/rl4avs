import gymnasium as gym
import highway_env
import time


def test_roundabout_rendering():
    """Quick test to ensure rendering works"""
    print("🧪 Testing roundabout visualization...")

    # Create environment with rendering
    env = gym.make("roundabout-v0", render_mode="human")

    # Run a few random steps
    obs, info = env.reset()

    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)

        env.render()
        time.sleep(0.1)

        if done or truncated:
            obs, info = env.reset()

    env.close()
    print("✅ Visualization test completed!")


if __name__ == "__main__":
    test_roundabout_rendering()
