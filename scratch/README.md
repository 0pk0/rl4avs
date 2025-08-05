# 🔧 Scratch Directory - Debugging & Testing Tools

This directory contains essential debugging and diagnostic scripts for the RL4AVS project. These tools help you quickly identify issues, test configurations, and optimize performance during development.

## 🧪 Available Tools

### 1. `debug_environment.py` - Environment Diagnostics
**Purpose**: Complete system health check for your RL environment setup.

**Features**:
- ✅ Dependency verification
- ✅ Environment creation testing
- ✅ Custom environment validation
- ✅ Model saving/loading verification
- ✅ Rendering capability testing

**Usage**:
```bash
python scratch/debug_environment.py
```

**When to use**: 
- First-time setup
- After installing new packages
- When encountering environment errors
- Before starting major experiments

---

### 2. `quick_train_test.py` - Fast Training Validation
**Purpose**: Rapid training tests with minimal timesteps for debugging.

**Features**:
- ⚡ Quick algorithm testing (1000 steps)
- 🧪 Multi-algorithm comparison
- 🏋️ Stress testing for stability
- 💾 Save/load verification
- 🎯 Prediction testing

**Usage**:
```bash
python scratch/quick_train_test.py
```

**When to use**:
- Testing new algorithm implementations
- Validating changes to training code
- Quick debugging before long training runs
- Checking algorithm stability

---

### 3. `test_model_loading.py` - Model Testing & Comparison
**Purpose**: Comprehensive testing of saved models and performance comparison.

**Features**:
- 🔍 Automatic model discovery
- 📊 Performance evaluation
- 🏆 Model comparison
- 📈 Success/collision rate analysis
- ✅ Loading verification

**Usage**:
```bash
python scratch/test_model_loading.py
```

**When to use**:
- After training sessions to validate results
- Comparing different algorithm performances
- Troubleshooting model loading issues
- Preparing for thesis result analysis

---

### 4. `performance_profiler.py` - Training Performance Analysis
**Purpose**: Deep performance analysis to identify bottlenecks and optimization opportunities.

**Features**:
- 📊 Real-time system monitoring (CPU, Memory)
- 🏃 Environment speed profiling
- 📈 FPS tracking during training
- 🔬 Comparative performance analysis
- 📊 Automatic performance visualization

**Usage**:
```bash
python scratch/performance_profiler.py
```

**When to use**:
- Optimizing training speed
- Identifying system bottlenecks
- Comparing environment configurations
- Preparing for large-scale experiments

---

## 🚀 Quick Start Workflow

### For New Setup:
1. **Environment Check**: `python scratch/debug_environment.py`
2. **Quick Validation**: `python scratch/quick_train_test.py`
3. **Performance Baseline**: `python scratch/performance_profiler.py`

### For Development:
1. **Code Changes**: `python scratch/quick_train_test.py` (test single algorithm)
2. **Performance Impact**: `python scratch/performance_profiler.py` (check speed)
3. **Model Validation**: `python scratch/test_model_loading.py` (verify results)

### For Debugging Issues:
1. **System Issues**: `python scratch/debug_environment.py`
2. **Training Issues**: `python scratch/quick_train_test.py` (stress test)
3. **Model Issues**: `python scratch/test_model_loading.py`

---

## 📁 Generated Files

The scratch directory will automatically create:
- `test_models/` - Quick test model saves
- `performance_plot.png` - Performance profiling visualizations
- `quick_test_*.zip` - Temporary model files from testing

**Note**: These files are temporary and safe to delete.

---

## 🔧 Customization

Each script supports customization:

```python
# Quick training with custom parameters
quick_train_test('PPO', timesteps=2000, use_custom_env=True)

# Performance profiling with different algorithms
profile_training_performance('DQN', timesteps=5000)

# Model evaluation with more episodes
evaluate_model_performance(model_path, model_name, n_episodes=10)
```

---

## 💡 Tips for MSc Research

1. **Before Major Experiments**: Always run `debug_environment.py` to ensure system stability
2. **Algorithm Development**: Use `quick_train_test.py` for rapid iteration
3. **Performance Optimization**: Use `performance_profiler.py` to optimize computational efficiency
4. **Result Validation**: Use `test_model_loading.py` for thesis-quality result verification
5. **Reproducibility**: All scripts use fixed seeds for consistent debugging

---

## 🆘 Troubleshooting

**Common Issues:**

| Issue | Solution |
|-------|----------|
| "No models found" | Train some models first with `experiments/train_agents.py` |
| Import errors | Run `debug_environment.py` to check dependencies |
| Slow performance | Use `performance_profiler.py` to identify bottlenecks |
| Rendering issues | Check display settings, run debug script |
| Save/load errors | Verify directory permissions and disk space |

**Getting Help:**
- Check the main project README.md
- Review individual script docstrings
- Use `python script_name.py --help` where available

---

**Happy Debugging! 🎯** 