import gymnasium as gym
import highway_env

import os
import json
import wandb  # Weights & Biases for experiment tracking
from datetime import datetime

# Initialize experiment tracking
wandb.login()  # You'll need to create a free account
wandb.init(project="roundabout-rl-msc", name="experiment-1")

# Create results directory structure
os.makedirs("results/experiments", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

env = gym.make("roundabout-v0", render_mode="human")  # For visualization
obs, info = env.reset()
done = truncated = False

while not (done or truncated):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
