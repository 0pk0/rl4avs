# 🚗 RL4AVS: Reinforcement Learning for Autonomous Vehicle Systems

A comprehensive MSc dissertation project implementing and comparing Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) algorithms for autonomous vehicle navigation in highway roundabout scenarios using the highway-env environment.

## 🎯 Project Overview

This project explores the application of various RL algorithms to train autonomous agents for safe and efficient roundabout navigation. The implementation focuses on both traditional RL algorithms (DQN, PPO, A2C) and advanced/niche algorithms for comprehensive comparison and analysis.

### 🏗️ Key Features

- **Multi-Algorithm Support**: PPO, DQN, A2C, and extensible for additional algorithms
- **Custom Environment Extensions**: Enhanced roundabout-v0 with custom reward shaping
- **Comprehensive Debugging**: Real-time action monitoring and visualization
- **Experiment Tracking**: TensorBoard integration and detailed logging
- **Performance Analysis**: Statistical comparison and visualization tools
- **Visual Training**: Real-time rendering during training and evaluation

## 📁 Project Structure

```
rl4avs/
├── src/                          # Core implementation
│   ├── agents.py                 # RL algorithm implementations
│   ├── environment.py            # Custom environment configurations
│   ├── evaluator.py              # Performance evaluation tools
│   ├── utils.py                  # Utility functions
│   └── visualization_callback.py # Training visualization and debugging
├── experiments/                  # Experiment scripts
│   ├── train_agents.py          # Enhanced training with debugging
│   ├── compare_algorithms.py    # Algorithm comparison
│   ├── watch_agents.py          # Trained agent visualization
│   ├── action_monitor.py        # Action distribution analysis
│   └── results/                 # Experiment-specific results
├── results/                      # Main results directory
│   ├── models/                  # Trained model files
│   ├── logs/                    # TensorBoard logs and JSON results
│   └── plots/                   # Generated visualizations
├── scratch/                      # Debugging and testing scripts
├── main.py                      # Basic environment testing
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rl4avs
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test installation**
   ```bash
   python main.py
   ```

## 🎮 Usage

### Basic Training

**Train PPO agent with debugging (recommended for development):**
```bash
cd experiments
python train_ppo.py
```

**Train DQN agent with debugging:**
```bash
cd experiments
python train_dqn.py
```

**Train A2C agent with debugging:**
```bash
cd experiments
python train_a2c.py
```

**Train multiple algorithms:**
```bash
cd experiments
python compare_algorithms.py
```

### Watching Trained Agents

**View a trained agent's performance:**
```bash
cd experiments
python watch_agents.py
```

### Action Analysis

**Monitor action distributions:**
```bash
cd experiments
python action_monitor.py
```

### Q-Learning Training

The `train_q_learning.py` script introduces a Q-Learning training framework for autonomous vehicle navigation, comparable to the existing PPO setup.

- **Environment and Action Space**: Utilizes the same custom environment (`roundabout-v0`) and action space as PPO, ensuring consistency in training conditions.
- **Evaluation and Metrics**: The evaluation framework is aligned with PPO, using the same episode limits, evaluation frequency, and custom reward functions. Generates comparable performance metrics for analysis.
- **Training Configuration**: Configurable parameters include total episodes, evaluation frequency, maximum steps per episode, and stopping mode. Enhanced reward features and Q-Learning hyperparameters are specified for effective training.
- **Q-Table and State Discretization**: Provides insights into the expected Q-table size and state discretization, crucial for understanding the agent's learning process.

### Training Parameters

**Default settings in `train_q_learning.py`:**
- Total episodes: 5,000 (equivalent to ~250K steps)
- Evaluation frequency: every 500 episodes
- Max steps per episode: 300
- Stopping mode: "safety" (same as PPO)
- Enhanced reward features: Strong idle penalty, high completion reward, efficiency bonus, etc.
- Q-Learning hyperparameters: Learning rate 0.1, discount factor 0.99, epsilon decay 0.995

### Debugging Tools

- **Environment Diagnostics**: `debug_environment.py` for system health checks
- **Quick Training Validation**: `quick_train_test.py` for rapid testing
- **Model Testing & Comparison**: `test_model_loading.py` for performance evaluation

### Algorithms Implemented

1. **PPO (Proximal Policy Optimization)**
   - Policy gradient method with clipped surrogate objective
   - Good for continuous action spaces
   - Stable and sample efficient
   - Enhanced with advanced hyperparameters and safety-based stopping criteria

2. **DQN (Deep Q-Network)**
   - Value-based method for discrete action spaces
   - Utilizes experience replay and target networks for stability
   - Integrated with safety-based stopping criteria for robust training

3. **A2C (Advantage Actor-Critic)**
   - Policy gradient method with value function baseline
   - On-policy learning that combines advantages of policy gradients and value-based methods
   - Integrated with safety-based stopping criteria for robust training

4. **Q-Learning**
   - Value-based method with Q-table updates
   - Suitable for discrete action spaces, using state discretization for continuous observations
   - Serves as a comparable baseline to deep RL methods within the safety-focused framework

### Planned Extensions

- **SAC (Soft Actor-Critic)**: For continuous control tasks
- **TD3 (Twin Delayed DDPG)**: Advanced policy gradient method
- **Rainbow DQN**: Enhanced DQN with multiple improvements
- **Custom algorithms**: Research-specific implementations

## 📈 Results and Analysis

### TensorBoard Monitoring

Launch TensorBoard to monitor training:
```bash
tensorboard --logdir=results/logs
```

### Performance Metrics

The system tracks multiple metrics:
- **Episode rewards**: Cumulative reward per episode
- **Episode length**: Number of steps before termination
- **Success rate**: Percentage of successful roundabout navigation
- **Collision rate**: Safety metric
- **Action distributions**: Behavioral analysis

### Statistical Analysis

Results include:
- Mean and standard deviation across seeds
- Learning curves and convergence analysis
- Comparative performance visualization
- Action frequency distributions

## 🐛 Debugging and Development

### Debugging Features

- **Real-time progress bars**: Visual training progress
- **Action frequency tracking**: Monitor agent behavior patterns
- **Reward trend analysis**: Identify learning issues
- **Visual evaluation**: See agent performance live
- **Comprehensive logging**: Detailed training statistics
- **Collision Insights**: Detailed analysis and solutions for collision issues

### Common Issues and Solutions

**Models not saving:**
- Check directory permissions
- Verify disk space
- Enable debug mode for detailed error messages

**Training performance issues:**
- Reduce total timesteps for testing
- Disable visual rendering for faster training
- Check environment configuration

**Environment rendering issues:**
- Ensure pygame is properly installed
- Check display forwarding for remote servers
- Use headless mode if visualization not needed

## 🔬 Research Extensions

### For Advanced Research

1. **Multi-Agent Scenarios**: Extend to multiple vehicles
2. **Transfer Learning**: Cross-domain knowledge transfer
3. **Meta-Learning**: Few-shot adaptation to new scenarios
4. **Curriculum Learning**: Progressive difficulty increase
5. **Safety Constraints**: Constrained RL for safety guarantees

### Custom Environment Modifications

The `CustomRoundaboutEnv` class provides hooks for:
- Custom reward functions
- Modified observation spaces
- Alternative action spaces
- Dynamic environment parameters

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Highway-Env Documentation](https://highway-env.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🤝 Contributing

This is an MSc dissertation project. For questions or suggestions:
1. Check existing issues and documentation
2. Test changes thoroughly
3. Follow existing code structure and style
4. Include proper documentation

## 📄 License

Academic use for MSc dissertation project.

## 🚨 Notes for MSc Dissertation

- **Experiment tracking**: All results automatically saved with timestamps
- **Reproducibility**: Fixed seeds and detailed configuration logging
- **Statistical significance**: Multiple seeds recommended for final results
- **Comprehensive analysis**: Use provided tools for thesis plots and tables
- **Performance optimization**: Monitor computational resources during training

---

**Last Updated**: August 2025
**Author**: Praveen Kathirvel - MSc Robotics
**Institution**: [University of Birmingham]

