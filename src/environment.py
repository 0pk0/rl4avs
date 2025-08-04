import gymnasium as gym
import highway_env
from highway_env.envs.roundabout_env import RoundaboutEnv
from gymnasium.envs.registration import register


class CustomRoundaboutEnv(RoundaboutEnv):
    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": True,
                "absolute": False,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [ 5, 10, 15, 20, 25, 30]
            },
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "duration": 20,
            "collision_reward": -10.0,
            "idle_penalty":-0.05,
            "high_speed_reward": 0.4,
            "arrived_reward": 3.0,
            "lane_change_reward": 0.1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "vehicles_count": 12,
        })
        return config

    def step(self, action):
        """Override step method to add custom reward modifications"""
        # Call the parent's step method first
        obs, reward, done, truncated, info = super().step(action)

        # ADD PENALTY FOR IDLING (low speed)
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.speed < 1.0:  # Less than 1 m/s is considered idling
                reward += self.config.get("idle_penalty", -0.05)

        # ADD REWARD FOR PROACTIVE LANE CHANGES
        # Action mapping: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
        if action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
            reward += self.config.get("lane_change_reward", 0.1)

        return obs, reward, done, truncated, info

def register_custom_env():
    """Register the custom environment"""
    try:
        register(
            id='custom-roundabout-v0',
            entry_point='src.environment:CustomRoundaboutEnv',
        )
        print("Custom environment registered successfully!")
    except Exception as e:
        print(f"Environment already registered or error: {e}")


def make_env(env_name="roundabout-v0", custom=False):
    """Create environment with optional customization"""
    if custom:
        register_custom_env()
        return gym.make('custom-roundabout-v0')
    else:
        return gym.make(env_name)
