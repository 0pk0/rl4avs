# 🚗 RL4AVS: Reinforcement Learning for Autonomous Vehicle Systems

A comprehensive MSc dissertation project implementing and comparing Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) algorithms for autonomous vehicle navigation in highway roundabout scenarios using the highway-env environment. The increasing complexity of autonomous driving scenarios necessitates robust and adaptive control strategies, making RL a promising avenue for research in this domain.

## 🎯 Project Overview

This project explores the application of various RL algorithms to train autonomous agents for safe and efficient roundabout navigation. The implementation focuses on both traditional RL algorithms (DQN, PPO, A2C) and advanced/niche algorithms for comprehensive comparison and analysis.

### 🏗️ Key Features

- **Multi-Algorithm Support**: PPO, DQN, A2C, and extensible for additional algorithms, facilitating broad comparative studies.
- **Custom Environment Extensions**: Enhanced roundabout-v0 with custom reward shaping, allowing for fine-grained control over experimental conditions and reward engineering, critical for evaluating algorithm performance in safety-critical applications.
- **Comprehensive Debugging**: Real-time action monitoring and visualization, aiding in understanding agent behavior and diagnosing learning issues.
- **Experiment Tracking**: TensorBoard integration and detailed logging, crucial for reproducible research and performance analysis.
- **Performance Analysis**: Statistical comparison and visualization tools, enabling robust evaluation of algorithm efficacy.
- **Visual Training**: Real-time rendering during training and evaluation, providing immediate qualitative feedback on agent performance.

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
├── projectResults/               # Evaluation results and TensorBoard logs for dissertation
├── results/                      # Main results directory
│   ├── models/                  # Trained model files
│   ├── logs/                    # TensorBoard logs and JSON results
│   └── plots/                   # Generated visualizations
├── scratch/                      # Debugging and testing scripts
├── main.py                      # Basic environment testing
└── requirements.txt             # Python dependencies
```
The modular design separates concerns into `src/` for core logic and `experiments/` for experimental setups, facilitating reproducibility and scalability of research.

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

1.  **PPO (Proximal Policy Optimization)**
    -   A policy gradient method leveraging a clipped surrogate objective to achieve stable and efficient learning. It is particularly well-suited for continuous action spaces, offering a balance between exploration and exploitation. Enhanced with advanced hyperparameters and safety-based stopping criteria to ensure robust training in safety-critical autonomous driving scenarios.
2.  **DQN (Deep Q-Network)**
    -   A value-based method for discrete action spaces that approximates the optimal action-value function using deep neural networks. It incorporates experience replay and target networks for improved stability. Integrated with safety-based stopping criteria for robust training, it provides a strong baseline for comparison.
3.  **A2C (Advantage Actor-Critic)**
    -   An on-policy actor-critic method that combines the strengths of policy gradients and value-based methods. It uses a value function baseline to reduce variance in policy gradient estimates, leading to more stable learning. Integrated with safety-based stopping criteria for robust training.
4.  **Q-Learning**
    -   A fundamental value-based reinforcement learning algorithm that learns an optimal action-selection policy by iteratively updating a Q-table. Suitable for discrete action spaces, it employs state discretization to handle continuous observations from the environment. Serves as a comparable baseline to deep RL methods within the safety-focused framework, providing insights into foundational RL performance.

### Evaluation and Statistical Metrics

To thoroughly evaluate the trained models and enable meaningful comparisons for the dissertation, a comprehensive set of evaluation and statistical metrics are employed. The evaluation framework is designed to provide insights into various aspects of agent performance, with a particular focus on safety, efficiency, quality, and behavioral characteristics. All evaluations are conducted against Q-Learning as a baseline model.

The metrics are logged to TensorBoard for visual trend analysis and saved to CSV/JSON for detailed statistical review. While efforts have been made to implement all desired metrics, some may not be fully functional due to complexities in environmental data extraction or time constraints within the project. The analysis will focus on the metrics that reliably provide data.

#### 🛡️ Safety Metrics

-   **Collision Rate**: The frequency of collisions per episode, a primary indicator of agent safety.
-   **Near-Miss Frequency**: The rate at which the ego vehicle comes dangerously close to other vehicles without actual collision. *(Note: This metric's implementation may be simplified due to environment limitations.)*
-   **Safety Margin Analysis**: The minimum distance maintained to other vehicles, reflecting the agent's cautiousness. *(Note: This metric's implementation may be simplified due to environment limitations.)*

#### ⚡ Efficiency Metrics

-   **Success Rate**: The percentage of episodes where the agent successfully navigates the roundabout and completes the course.
-   **Mean Episode Length**: The average number of steps taken to complete an episode, indicating efficiency.
-   **Completion Time Statistics**: Analysis of the time taken to complete successful episodes.

#### 📈 Quality Metrics

-   **Mean Episodic Reward**: The average total reward accumulated per episode, reflecting overall performance.
-   **Performance Consistency**: Measured by the standard deviation of episodic rewards or success rates over time, indicating the reliability of the agent's performance.
-   **Comfort Scores**: An indicator of the smoothness and human-like nature of the agent's driving behavior, often inversely proportional to acceleration magnitudes. *(Note: This metric's implementation may be simplified due to environment limitations.)*

#### 🧠 Behavioral Metrics

-   **Action Distribution Analysis**: The frequency and patterns of actions taken by the agent, offering insights into its decision-making process.
-   **Decision-Making Patterns**: Qualitative analysis of how the agent chooses actions in various scenarios, particularly in complex or risky situations. *(Note: This metric's implementation may be simplified due to environment limitations.)*

All metrics are plotted against each other on the same graph within TensorBoard, allowing for direct visual comparison of model performance trends over evaluation episodes. Convergence is observed by analyzing the flattening of performance curves for smoothed metrics like episodic reward and success rate.

### Planned Extensions

-   **SAC (Soft Actor-Critic)**: Considered for its state-of-the-art performance and sample efficiency in continuous control tasks, which is highly relevant to advanced autonomous driving scenarios.
-   **TD3 (Twin Delayed DDPG)**: An advanced policy gradient method known for addressing overestimation bias in Q-learning and improving stability in continuous action spaces.
-   **Rainbow DQN**: An enhancement of DQN that combines several improvements (e.g., Double DQN, Prioritized Experience Replay, Dueling Networks) to boost performance and stability.
-   **Custom algorithms**: Provision for research-specific implementations to explore novel approaches to autonomous vehicle control.

## 📈 Results and Analysis

### TensorBoard Monitoring

Launch TensorBoard to monitor training:
```bash
tensorboard --logdir=results/logs
```

### Performance Metrics

The system tracks multiple metrics, which directly inform the evaluation of research questions and hypotheses:
-   **Episode rewards**: Cumulative reward per episode, indicating overall agent performance.
-   **Episode length**: Number of steps before termination, reflecting efficiency and task completion.
-   **Success rate**: Percentage of successful roundabout navigation, a key safety and performance metric.
-   **Collision rate**: Safety metric, indicating the frequency of unsafe interactions.
-   **Action distributions**: Behavioral analysis, providing insights into the agent's decision-making process.

### Statistical Analysis

Results include:
-   Mean and standard deviation across seeds, crucial for assessing result robustness.
-   Learning curves and convergence analysis, illustrating training dynamics.
-   Comparative performance visualization, enabling clear comparison between algorithms.
-   Action frequency distributions, providing a detailed view of learned policies.
Statistical tests (e.g., t-tests, ANOVA) are employed to compare algorithm performance and confirm statistical significance.

## 🐛 Debugging and Development

### Debugging Features

-   **Real-time progress bars**: Visual training progress, enhancing user experience and immediate feedback.
-   **Action frequency tracking**: Monitor agent behavior patterns, aiding in understanding the learning process and diagnosing RL-specific issues.
-   **Reward trend analysis**: Identify learning issues, providing insights into reward function effectiveness.
-   **Visual evaluation**: See agent performance live, for qualitative assessment of learned policies.
-   **Comprehensive logging**: Detailed training statistics, supporting in-depth analysis.
-   **Collision Insights**: Detailed analysis and solutions for collision issues, critical for safety-focused development.

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

1.  **Multi-Agent Scenarios**: Extend to multiple vehicles interacting in the environment, addressing complex coordination and emergent behaviors crucial for real-world autonomous systems.
2.  **Transfer Learning**: Investigate cross-domain knowledge transfer to leverage pre-trained models or policies from simpler environments to more complex ones, accelerating training and improving performance.
3.  **Meta-Learning**: Explore few-shot adaptation to new scenarios or variations in the environment, enabling agents to quickly learn and generalize to unseen conditions.
4.  **Curriculum Learning**: Implement progressive difficulty increase in training scenarios, optimizing the learning process by starting with simpler tasks and gradually introducing complexity.
5.  **Safety Constraints**: Incorporate constrained RL techniques to ensure safety guarantees during autonomous vehicle operation, a paramount concern for deployment.

### Custom Environment Modifications

The `CustomRoundaboutEnv` class provides hooks for:
-   Custom reward functions, enabling precise shaping of agent behavior.
-   Modified observation spaces, allowing for exploration of different sensory inputs.
-   Alternative action spaces, facilitating research into diverse control strategies.
-   Dynamic environment parameters, supporting robust testing under varying conditions.

## 📚 References

-   [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
-   [Highway-Env Documentation](https://highway-env.readthedocs.io/)
-   [Gymnasium Documentation](https://gymnasium.farama.org/)
-   [**PPO** - Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.](https://arxiv.org/abs/1707.06347)
-   [**DQN** - Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533.](https://www.nature.com/articles/nature14236)
-   [**A2C** - Mnih, V., Badia, A. P., Babuschkin, I., Fortunato, M., Kurel, D., Leibo, Z. I., & Silver, D. (2016). Asynchronous methods for deep reinforcement learning. *International Conference on Machine Learning*.](http://proceedings.mlr.press/v48/mnih16.pdf)
-   [**Q-Learning** - Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine learning, 8*(3-4), 279-292.](https://link.springer.com/article/10.1007/BF00992698)



## 📄 License

Academic use for MSc dissertation project.

## 🚨 Notes for MSc Dissertation

-   **Experiment tracking**: All results automatically saved with timestamps, ensuring the high standard of reproducibility and traceability required for academic research.
-   **Reproducibility**: Fixed seeds and detailed configuration logging, paramount for validating experimental findings and facilitating future research.
-   **Statistical significance**: Multiple seeds recommended for final results, enabling robust statistical analysis and confident conclusions.
-   **Comprehensive analysis**: Use provided tools for thesis plots and tables, supporting rigorous data presentation and interpretation.
-   **Performance optimization**: Monitor computational resources during training, essential for efficient resource management and large-scale experiments.

---

**Last Updated**: 19 August 2025
**Author**: Praveen Kathirvel - MSc Robotics
**Institution**: [University of Birmingham]

