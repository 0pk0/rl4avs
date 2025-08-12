import gymnasium as gym
import highway_env
from highway_env.envs.roundabout_env import RoundaboutEnv
from gymnasium.envs.registration import register


class CustomRoundaboutEnv(RoundaboutEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config, render_mode=render_mode)
        self._init_tracking_variables()
        
    def _init_tracking_variables(self):
        """Initialize tracking variables for reward system"""
        # Track agent progress and mistakes
        self.prev_position = None
        self.stationary_steps = 0
        self.total_distance_traveled = 0
        self.mistake_memory = []  # Track recent collision/failure patterns
        self.max_stationary_steps = 10  # Max allowed stationary steps before heavy penalty
        self.step_count = 0  # Track steps for efficiency bonus
        
        # Course completion tracking
        self.initial_position = None
        self.has_entered_roundabout = False
        self.has_exited_roundabout = False
        self.roundabout_progress = 0
        self.completion_threshold = 150.0  # Distance needed to complete course
        self.exit_confirmation_threshold = 200.0  # Distance to confirm full exit
        self.max_distance_from_start = 0
        self.roundabout_center_distance = 0  # Track distance from roundabout center

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
            "duration": 200,  # Much longer episodes for course completion
            "collision_reward": -8.0,         # Strong but learnable collision penalty
            "idle_penalty": -0.3,             # Strong idle penalty but not overwhelming
            "stationary_penalty": -1.0,       # Heavy penalty for staying still
            "progress_reward": 0.2,           # Reward for making progress
            "high_speed_reward": 0.4,         # Moderate speed reward
            "arrived_reward": 15.0,           # Very high completion reward!
            "lane_change_reward": 0.15,       # Moderate lane change reward
            "efficiency_bonus": 3.0,          # Higher bonus for fast completion
            "repeated_mistake_penalty": -3.0,  # Moderate penalty for repeated mistakes
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "vehicles_count": 12,
        })
        return config

    def step(self, action):
        """Enhanced step method with aggressive reward shaping for course completion"""
        # Call the parent's step method first
        obs, reward, done, truncated, info = super().step(action)
        self.step_count += 1  # Track steps for efficiency bonus

        if hasattr(self, 'vehicle') and self.vehicle is not None:
            current_position = (self.vehicle.position[0], self.vehicle.position[1])
            current_speed = self.vehicle.speed
            
            # Initialize starting position on first step
            if self.initial_position is None:
                self.initial_position = current_position
                print(f"🏁 Starting position: ({current_position[0]:.1f}, {current_position[1]:.1f})")
            
            # 🎯 ENHANCED ROUNDABOUT EXIT DETECTION
            distance_from_start = abs(current_position[0] - self.initial_position[0]) + \
                                abs(current_position[1] - self.initial_position[1])
            
            # Track maximum distance traveled (progress indicator)
            if distance_from_start > self.max_distance_from_start:
                self.max_distance_from_start = distance_from_start
            
            # Estimate roundabout center distance (assuming roundabout is around initial position)
            # This helps detect when vehicle is moving away from roundabout
            roundabout_center = self.initial_position  # Approximate center
            center_distance = abs(current_position[0] - roundabout_center[0]) + \
                            abs(current_position[1] - roundabout_center[1])
            self.roundabout_center_distance = center_distance
            
            # 🔄 ROUNDABOUT ENTRY DETECTION
            if not self.has_entered_roundabout and distance_from_start > 50.0:
                self.has_entered_roundabout = True
                print(f"🔄 ENTERED ROUNDABOUT at step {self.step_count}")
                print(f"   Entry position: ({current_position[0]:.1f}, {current_position[1]:.1f})")
                print(f"   Entry speed: {current_speed:.1f}")
            
            # 🚪 ROUNDABOUT EXIT DETECTION 
            # More sophisticated detection: vehicle has traveled far AND is moving away from roundabout
            if (self.has_entered_roundabout and 
                not self.has_exited_roundabout and
                distance_from_start > self.completion_threshold and
                center_distance > self.completion_threshold * 0.8 and  # Moving away from center
                self.step_count > 30):  # Minimum steps for legitimate exit
                
                # Confirm vehicle is actively moving (not crashed/stuck)
                if current_speed > 5.0:
                    self.has_exited_roundabout = True
                    print(f"🚪 ROUNDABOUT EXIT DETECTED at step {self.step_count}")
                    print(f"   Exit position: ({current_position[0]:.1f}, {current_position[1]:.1f})")
                    print(f"   Exit speed: {current_speed:.1f}")
                    print(f"   Distance from start: {distance_from_start:.1f}")
                    print(f"   Distance from roundabout center: {center_distance:.1f}")
            
            # 🏆 FINAL COMPLETION DETECTION: Vehicle has fully exited and traveled sufficient distance
            if (self.has_exited_roundabout and 
                distance_from_start > self.exit_confirmation_threshold and
                current_speed > 3.0):  # Still moving after exit
                
                print(f"")
                print(f"🎉 =============================================")
                print(f"🎉    ROUNDABOUT COURSE COMPLETED!          ")
                print(f"🎉 =============================================")
                print(f"📊 COMPLETION SUMMARY:")
                print(f"   ✅ Total distance traveled: {distance_from_start:.1f} units")
                print(f"   ⏱️  Total steps taken: {self.step_count}")
                print(f"   🏁 Final position: ({current_position[0]:.1f}, {current_position[1]:.1f})")
                print(f"   🚗 Final speed: {current_speed:.1f} km/h")
                print(f"   🔄 Roundabout entry: Step {3}")
                print(f"   🚪 Roundabout exit: Step ~{self.step_count-20}")
                print(f"   🏆 Course completion: Step {self.step_count}")
                print(f"🎉 =============================================")
                print(f"")
                
                # Set arrival flag and force immediate termination
                info['arrived'] = True
                done = True  # Force episode termination immediately
                
                # Log this like a collision for debugging
                print(f"💫 EPISODE TERMINATED: Course completion detected")
                print(f"   Termination reason: Successful roundabout navigation")
            
            # 🚨 AGGRESSIVE IDLE/STATIONARY PENALTIES
            if current_speed < 0.5:  # Very low speed threshold
                # Progressive idle penalty - gets worse over time
                idle_penalty = self.config.get("idle_penalty", -0.5)
                reward += idle_penalty
                self.stationary_steps += 1
                
                # HEAVY penalty for staying still too long
                if self.stationary_steps > self.max_stationary_steps:
                    stationary_penalty = self.config.get("stationary_penalty", -2.0)
                    # Exponentially increasing penalty
                    multiplier = min(self.stationary_steps - self.max_stationary_steps, 10)
                    reward += stationary_penalty * multiplier
                    
                    if self.stationary_steps > 20:  # Force action after 20 steps of being still
                        reward -= 10.0  # Massive penalty
                        
            else:
                self.stationary_steps = 0  # Reset counter when moving
                
            # 🎯 PROGRESS REWARDS - Encourage forward movement
            if self.prev_position is not None:
                distance_moved = abs(current_position[0] - self.prev_position[0]) + \
                               abs(current_position[1] - self.prev_position[1])
                self.total_distance_traveled += distance_moved
                
                if distance_moved > 0.1:  # Reward any meaningful movement
                    progress_reward = self.config.get("progress_reward", 0.3)
                    reward += progress_reward * distance_moved
            
            self.prev_position = current_position
            
            # 🏆 MASSIVE COMPLETION REWARD
            if info.get('arrived', False):
                arrival_reward = self.config.get("arrived_reward", 15.0)
                reward += arrival_reward
                
                # 🚀 EFFICIENCY BONUS - Extra reward for fast completion
                if self.step_count < 150:  # Reasonable completion time
                    efficiency_bonus = self.config.get("efficiency_bonus", 3.0)
                    speed_multiplier = max(1.0, (150 - self.step_count) / 100)
                    reward += efficiency_bonus * speed_multiplier
                    print(f"   🚀 Efficiency bonus: +{efficiency_bonus * speed_multiplier:.2f}")
                
                # 🏃 DISTANCE BONUS - Reward for traveling sufficient distance
                distance_bonus = min(5.0, self.max_distance_from_start / 50.0)
                reward += distance_bonus
                print(f"   🏃 Distance bonus: +{distance_bonus:.2f}")
                
                print(f"🎉 COURSE COMPLETED! Total reward this episode: {reward:.2f}")
                print(f"   🏆 This should be the MAXIMUM reward possible!")
            
            # 💥 ENHANCED COLLISION PENALTY with mistake tracking
            if info.get('crashed', False):
                collision_penalty = self.config.get("collision_reward", -15.0)
                reward += collision_penalty
                
                # Track mistake patterns to prevent repetition
                mistake_info = {
                    'position': current_position,
                    'speed': current_speed,
                    'action': action
                }
                self.mistake_memory.append(mistake_info)
                
                # Additional penalty for repeated mistakes at similar locations
                if len(self.mistake_memory) > 1:
                    for prev_mistake in self.mistake_memory[-3:]:  # Check last 3 mistakes
                        pos_diff = abs(current_position[0] - prev_mistake['position'][0]) + \
                                  abs(current_position[1] - prev_mistake['position'][1])
                        if pos_diff < 2.0:  # Similar location
                            repeated_penalty = self.config.get("repeated_mistake_penalty", -5.0)
                            reward += repeated_penalty
                            break
                
                print(f"")
                print(f"💥 =============================================")
                print(f"💥    COLLISION DETECTED!                   ")
                print(f"💥 =============================================")
                print(f"📊 COLLISION SUMMARY:")
                print(f"   💥 Collision position: ({current_position[0]:.1f}, {current_position[1]:.1f})")
                print(f"   ⏱️  Steps before collision: {self.step_count}")
                print(f"   🚗 Speed at impact: {current_speed:.1f} km/h")
                print(f"   🔄 Had entered roundabout: {self.has_entered_roundabout}")
                print(f"   🚪 Had exited roundabout: {self.has_exited_roundabout}")
                print(f"   📏 Distance from start: {abs(current_position[0] - self.initial_position[0]) + abs(current_position[1] - self.initial_position[1]):.1f}")
                print(f"   🎯 Penalty applied: {collision_penalty:.1f}")
                print(f"   📊 Episode reward: {reward:.2f}")
                print(f"💥 =============================================")
                print(f"")
                print(f"💫 EPISODE TERMINATED: Collision detected")
                print(f"   Termination reason: Vehicle collision")
            
            # 🚗 SPEED REWARDS - Encourage appropriate speed
            if 8.0 <= current_speed <= 15.0:  # Optimal speed range for roundabouts
                speed_reward = self.config.get("high_speed_reward", 0.6)
                reward += speed_reward
            elif current_speed > 20.0:  # Penalize excessive speed
                reward -= 0.3
            
            # 🔄 LANE CHANGE REWARDS (enhanced)
            if action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
                lane_reward = self.config.get("lane_change_reward", 0.2)
                reward += lane_reward
            
            # 🚫 DISCOURAGE IDLE ACTION
            if action == 1:  # IDLE action
                reward -= 0.2  # Small penalty for choosing to do nothing
            
            # 🏃 ENCOURAGE SPEED ACTIONS
            if action in [3]:  # FASTER action
                reward += 0.1  # Small reward for acceleration
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset tracking variables"""
        obs, info = super().reset(**kwargs)
        # Ensure tracking variables are initialized
        if not hasattr(self, 'mistake_memory'):
            self._init_tracking_variables()
        else:
            self.prev_position = None
            self.stationary_steps = 0
            self.total_distance_traveled = 0
            self.mistake_memory = self.mistake_memory[-5:] if len(self.mistake_memory) > 5 else []  # Keep recent mistakes
            self.step_count = 0
            # Reset completion tracking
            self.initial_position = None
            self.has_entered_roundabout = False
            self.has_exited_roundabout = False
            self.roundabout_progress = 0
            self.max_distance_from_start = 0
            self.roundabout_center_distance = 0
        return obs, info

def register_custom_env():
    """Register the custom environment"""
    try:
        register(
            id='custom-roundabout-v0',
            entry_point='src.environment:CustomRoundaboutEnv',
        )
        print("Custom environment registered successfully!")
    except Exception as e:
        # This is expected on subsequent calls
        if "already in registry" not in str(e):
            print(f"Environment registration error: {e}")


def make_env(env_name="roundabout-v0", custom=False):
    """Create environment with optional customization"""
    if custom:
        register_custom_env()
        return gym.make('custom-roundabout-v0')
    else:
        return gym.make(env_name)
