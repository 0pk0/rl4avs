#!/usr/bin/env python3
"""
Performance profiling script for RL training
Helps identify bottlenecks and optimize training speed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import psutil
import gymnasium as gym
from stable_baselines3 import PPO
from src.environment import make_env
import threading
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class PerformanceProfiler:
    """Monitor system performance during training"""
    
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.timestamps = deque()
        self.cpu_percent = deque()
        self.memory_percent = deque()
        self.memory_mb = deque()
        self.fps_data = deque()
        
        self.start_time = None
        self.step_count = 0
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("📊 Performance monitoring started...")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("📊 Performance monitoring stopped.")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            current_time = time.time() - self.start_time
            
            # System metrics
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Store data
            self.timestamps.append(current_time)
            self.cpu_percent.append(cpu)
            self.memory_percent.append(memory.percent)
            self.memory_mb.append(memory.used / 1024 / 1024)
            
            # Calculate FPS if we have step data
            if current_time > 0:
                fps = self.step_count / current_time
                self.fps_data.append(fps)
            else:
                self.fps_data.append(0)
            
            time.sleep(self.monitor_interval)
    
    def update_steps(self, steps):
        """Update step count for FPS calculation"""
        self.step_count = steps
    
    def get_summary(self):
        """Get performance summary"""
        if not self.timestamps:
            return "No data collected"
        
        total_time = self.timestamps[-1]
        avg_cpu = np.mean(self.cpu_percent)
        max_cpu = np.max(self.cpu_percent)
        avg_memory = np.mean(self.memory_mb)
        max_memory = np.max(self.memory_mb)
        avg_fps = np.mean(self.fps_data) if self.fps_data else 0
        
        summary = f"""
Performance Summary:
==================
Total time: {total_time:.1f} seconds
Steps: {self.step_count}
Average FPS: {avg_fps:.1f}

CPU Usage:
  Average: {avg_cpu:.1f}%
  Peak: {max_cpu:.1f}%

Memory Usage:
  Average: {avg_memory:.0f} MB
  Peak: {max_memory:.0f} MB
        """
        return summary
    
    def plot_performance(self, save_path="scratch/performance_plot.png"):
        """Create performance visualization"""
        if not self.timestamps:
            print("No data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Performance Profile')
        
        times = list(self.timestamps)
        
        # CPU usage
        ax1.plot(times, list(self.cpu_percent), 'b-', alpha=0.7)
        ax1.set_title('CPU Usage (%)')
        ax1.set_ylabel('CPU %')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        ax2.plot(times, list(self.memory_mb), 'r-', alpha=0.7)
        ax2.set_title('Memory Usage (MB)')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True, alpha=0.3)
        
        # FPS
        ax3.plot(times, list(self.fps_data), 'g-', alpha=0.7)
        ax3.set_title('Training Speed (FPS)')
        ax3.set_ylabel('Steps/Second')
        ax3.set_xlabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Memory percentage
        ax4.plot(times, list(self.memory_percent), 'm-', alpha=0.7)
        ax4.set_title('Memory Usage (%)')
        ax4.set_ylabel('Memory %')
        ax4.set_xlabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Performance plot saved to {save_path}")

def profile_environment_speed():
    """Profile environment step speed"""
    print("🏃 Profiling Environment Speed")
    print("-" * 40)
    
    environments = [
        ("Standard Roundabout", lambda: gym.make("roundabout-v0")),
        ("Custom Roundabout", lambda: make_env("roundabout-v0", custom=True)),
        ("Standard with Render", lambda: gym.make("roundabout-v0", render_mode="human"))
    ]
    
    results = {}
    
    for env_name, env_factory in environments:
        try:
            print(f"Testing {env_name}...")
            env = env_factory()
            
            # Warmup
            obs, info = env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    obs, info = env.reset()
            
            # Timing test
            n_steps = 1000
            start_time = time.time()
            
            for i in range(n_steps):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    obs, info = env.reset()
            
            elapsed_time = time.time() - start_time
            fps = n_steps / elapsed_time
            
            env.close()
            
            results[env_name] = {
                'fps': fps,
                'step_time_ms': elapsed_time / n_steps * 1000
            }
            
            print(f"  FPS: {fps:.1f}")
            print(f"  Step time: {elapsed_time / n_steps * 1000:.2f} ms")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[env_name] = None
    
    print("\n📊 Environment Speed Summary:")
    for env_name, result in results.items():
        if result:
            print(f"  {env_name}: {result['fps']:.1f} FPS")
    
    return results

def profile_training_performance(algorithm='PPO', timesteps=5000):
    """Profile training performance with system monitoring"""
    print(f"🔬 Profiling {algorithm} Training Performance")
    print(f"Timesteps: {timesteps}")
    print("-" * 40)
    
    # Initialize profiler
    profiler = PerformanceProfiler(monitor_interval=0.5)
    
    try:
        # Create environment and model
        env = make_env("roundabout-v0", custom=False)
        model = PPO('MlpPolicy', env, verbose=0)
        
        # Start monitoring
        profiler.start_monitoring()
        
        # Training with step tracking
        print("🚀 Starting training...")
        start_time = time.time()
        
        # Custom training loop for step tracking
        model.learn(total_timesteps=timesteps, progress_bar=True)
        
        training_time = time.time() - start_time
        profiler.update_steps(timesteps)
        
        # Stop monitoring
        profiler.stop_monitoring()
        
        # Results
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        print(f"📊 Overall FPS: {timesteps / training_time:.1f}")
        
        # Detailed performance summary
        print(profiler.get_summary())
        
        # Create performance plot
        profiler.plot_performance()
        
        env.close()
        
        return {
            'total_time': training_time,
            'fps': timesteps / training_time,
            'profiler': profiler
        }
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        profiler.stop_monitoring()
        return None

def compare_training_speeds():
    """Compare training speeds across different configurations"""
    print("🏁 Training Speed Comparison")
    print("=" * 50)
    
    configurations = [
        ("PPO Default", lambda: PPO('MlpPolicy', make_env("roundabout-v0"), verbose=0)),
        ("PPO Custom Env", lambda: PPO('MlpPolicy', make_env("roundabout-v0", custom=True), verbose=0)),
        ("DQN Default", lambda: None),  # Skip DQN for speed
    ]
    
    timesteps = 2000  # Reduced for quick comparison
    results = {}
    
    for config_name, model_factory in configurations:
        if model_factory() is None:
            continue
            
        print(f"\n🧪 Testing {config_name}")
        try:
            model = model_factory()
            
            start_time = time.time()
            model.learn(total_timesteps=timesteps, progress_bar=False)
            training_time = time.time() - start_time
            
            fps = timesteps / training_time
            results[config_name] = {
                'time': training_time,
                'fps': fps
            }
            
            print(f"   Time: {training_time:.1f}s, FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary
    print("\n📊 Speed Comparison:")
    if results:
        fastest = max(results.items(), key=lambda x: x[1]['fps'])
        print(f"🏆 Fastest: {fastest[0]} ({fastest[1]['fps']:.1f} FPS)")
        
        for config, result in results.items():
            relative_speed = result['fps'] / fastest[1]['fps'] * 100
            print(f"   {config}: {result['fps']:.1f} FPS ({relative_speed:.0f}%)")

def main():
    """Main profiling interface"""
    print("🔬 Performance Profiler")
    print("=" * 30)
    
    tests = [
        ("Environment Speed", profile_environment_speed),
        ("Training Performance", lambda: profile_training_performance('PPO', 3000)),
        ("Speed Comparison", compare_training_speeds),
    ]
    
    print("Available tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    
    try:
        choice = input("\nChoose test (1-3) or press Enter for all: ").strip()
        
        if choice == "":
            # Run all tests
            for name, test_func in tests:
                print(f"\n{'='*20} {name} {'='*20}")
                test_func()
        elif choice.isdigit() and 1 <= int(choice) <= len(tests):
            name, test_func = tests[int(choice) - 1]
            print(f"\n{'='*20} {name} {'='*20}")
            test_func()
        else:
            print("Invalid choice, running training performance test...")
            profile_training_performance('PPO', 2000)
            
    except KeyboardInterrupt:
        print("\n⚠️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
    
    print("\n🏁 Profiling completed!")
    print("💡 Check scratch/ directory for performance plots")

if __name__ == "__main__":
    main() 