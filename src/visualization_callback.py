import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import time
from tqdm import tqdm
from collections import defaultdict, deque


class VisualizationCallbackWithDebug(BaseCallback):
    """
    Enhanced callback with progress tracking and action debugging
    """

    def __init__(self, eval_env, eval_freq=10000, render_freq=5000,
                 total_timesteps=10000, debug_actions=True, verbose=1):
        super(VisualizationCallbackWithDebug, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.render_freq = render_freq
        self.eval_episodes = 3
        self.total_timesteps = total_timesteps
        self.debug_actions = debug_actions

        # Progress tracking
        self.pbar = None
        self.last_update = 0

        # Action debugging
        self.action_counts = defaultdict(int)
        self.recent_actions = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        self.debug_frequency = 1000  # Debug every 1000 steps

        # Action mapping for highway-env roundabout
        self.action_names = {
            0: "LANE_LEFT",
            1: "IDLE",
            2: "LANE_RIGHT",
            3: "FASTER",
            4: "SLOWER"
        }

    def _on_training_start(self) -> None:
        """Initialize progress bar when training starts"""
        print("🚀 Starting training with action debugging enabled!")
        print("Action meanings:")
        for action_id, name in self.action_names.items():
            print(f"  {action_id}: {name}")
        print("-" * 50)

        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="🤖 Training Agent",
            unit="steps",
            ncols=120,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

    def _on_step(self) -> bool:
        # Extract current action and reward
        current_action = None
        current_reward = None

        # Get action from the locals (stable-baselines3 stores this)
        if 'actions' in self.locals:
            current_action = self.locals['actions']
            if hasattr(current_action, 'item'):  # Convert tensor to scalar
                current_action = current_action.item()
            elif isinstance(current_action, (list, np.ndarray)):
                current_action = current_action[0]

        # Get reward
        if 'rewards' in self.locals:
            current_reward = self.locals['rewards']
            if hasattr(current_reward, 'item'):
                current_reward = current_reward.item()
            elif isinstance(current_reward, (list, np.ndarray)):
                current_reward = current_reward[0]

        # Log action and reward
        if current_action is not None:
            self.action_counts[current_action] += 1
            self.recent_actions.append(current_action)

        if current_reward is not None:
            self.recent_rewards.append(current_reward)

        # Update progress bar
        if self.pbar:
            steps_since_last = self.n_calls - self.last_update
            self.pbar.update(steps_since_last)
            self.last_update = self.n_calls

            # Update progress bar with recent metrics
            postfix_dict = {}
            if len(self.recent_rewards) > 0:
                postfix_dict['Avg_Reward'] = f"{np.mean(list(self.recent_rewards)):.2f}"
            if len(self.recent_actions) > 0:
                most_common_action = max(set(self.recent_actions), key=list(self.recent_actions).count)
                postfix_dict['Common_Action'] = self.action_names.get(most_common_action, str(most_common_action))

            self.pbar.set_postfix(postfix_dict)

        # Debug actions periodically
        if self.debug_actions and self.n_calls % self.debug_frequency == 0:
            self._debug_actions()

        # Show live training visualization every render_freq steps
        if self.n_calls % self.render_freq == 0:
            self._render_current_episode()

        # Detailed evaluation with visualization every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_with_visualization()

        return True

    def _debug_actions(self):
        """Print detailed action debugging information"""
        if self.pbar:
            self.pbar.write(f"\n📊 Action Debug at step {self.n_calls:,}")

            # Action distribution
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                self.pbar.write("Action Distribution:")
                for action_id, count in sorted(self.action_counts.items()):
                    percentage = (count / total_actions) * 100
                    action_name = self.action_names.get(action_id, f"Unknown_{action_id}")
                    self.pbar.write(f"  {action_name}: {count} ({percentage:.1f}%)")

            # Recent action sequence
            if len(self.recent_actions) >= 10:
                recent_sequence = list(self.recent_actions)[-10:]
                action_sequence = [self.action_names.get(a, str(a)) for a in recent_sequence]
                self.pbar.write(f"Last 10 actions: {' → '.join(action_sequence)}")

            # Performance metrics
            if len(self.recent_rewards) > 0:
                avg_reward = np.mean(list(self.recent_rewards))
                min_reward = min(self.recent_rewards)
                max_reward = max(self.recent_rewards)
                self.pbar.write(f"Recent rewards - Avg: {avg_reward:.3f}, Min: {min_reward:.3f}, Max: {max_reward:.3f}")
                idle_pct = self.action_counts.get(1, 0) / max(1, sum(self.action_counts.values()))
                slower_pct = self.action_counts.get(4, 0) / max(1, sum(self.action_counts.values()))
                if idle_pct > 0.30 or slower_pct > 0.30:
                    self.pbar.write(
                        "⚠️  High IDLE/SLOWER usage detected "
                        f"(IDLE {idle_pct:.1%}, SLOWER {slower_pct:.1%}). "
                        "Consider increasing exploration or adjusting rewards."
                    )
            self.pbar.write("-" * 50)

    def _on_training_end(self) -> None:
        """Close progress bar and show final action summary"""
        if self.pbar:
            self.pbar.write("\n🎉 Training completed!")
            self._print_final_action_summary()
            self.pbar.close()

    def _print_final_action_summary(self):
        """Print comprehensive action summary at the end of training"""
        print("\n" + "=" * 60)
        print("FINAL ACTION ANALYSIS")
        print("=" * 60)

        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            print("Overall Action Distribution:")
            for action_id, count in sorted(self.action_counts.items()):
                percentage = (count / total_actions) * 100
                action_name = self.action_names.get(action_id, f"Unknown_{action_id}")
                bar_length = int(percentage / 2)  # Scale for terminal display
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {action_name:12}: {bar} {percentage:5.1f}% ({count:,} times)")

        if len(self.recent_rewards) > 0:
            print(f"\nFinal Performance:")
            print(f"  Average recent reward: {np.mean(list(self.recent_rewards)):.3f}")
            print(
                f"  Reward trend: {np.mean(list(self.recent_rewards)[-20:]) - np.mean(list(self.recent_rewards)[:20]):.3f}")

        print("=" * 60)

    def _render_current_episode(self):
        """Show current training episode"""
        if hasattr(self.training_env, 'render'):
            try:
                self.training_env.render()
                time.sleep(0.1)
            except:
                pass

    def _evaluate_with_visualization(self):
        """Run evaluation episodes with full visualization and action logging"""
        if self.pbar:
            self.pbar.write(f"\n🎬 Detailed evaluation at step {self.n_calls:,}")

        eval_rewards = []
        eval_actions = []

        # Create evaluation progress bar
        eval_pbar = tqdm(
            range(self.eval_episodes),
            desc="🎭 Evaluation",
            leave=False,
            ncols=80
        )

        for episode in eval_pbar:
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            done = truncated = False

            eval_pbar.set_description(f"🎭 Eval Episode {episode + 1}")

            while not (done or truncated):
                # Get action from current policy
                action, _ = self.model.predict(obs, deterministic=True)
                if hasattr(action, 'item'):
                    action_value = action.item()
                else:
                    action_value = action

                episode_actions.append(action_value)

                obs, reward, done, truncated, info = self.eval_env.step(action)

                # Render the environment
                self.eval_env.render()
                time.sleep(0.05)

                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_actions.extend(episode_actions)

            # Episode summary with action analysis
            action_counts = defaultdict(int)
            for a in episode_actions:
                action_counts[a] += 1

            most_used_action = max(action_counts, key=action_counts.get) if action_counts else 0
            action_name = self.action_names.get(most_used_action, str(most_used_action))

            outcome = "❌ Collision" if info.get('crashed', False) else "✅ Success" if info.get('arrived',
                                                                                               False) else "⏱️ Timeout"

            if self.pbar:
                self.pbar.write(f"    {outcome} | Reward: {episode_reward:.2f} | "
                                f"Length: {episode_length} | Most used: {action_name}")

        eval_pbar.close()

        # Evaluation summary
        if self.pbar:
            avg_eval_reward = np.mean(eval_rewards)
            self.pbar.write(f"📈 Evaluation average reward: {avg_eval_reward:.3f}")
