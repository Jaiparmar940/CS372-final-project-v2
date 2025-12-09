# Rocket Lander RL Project

A comprehensive reinforcement learning project comparing Deep Q-Network (DQN) and Actor-Critic (A2C) algorithms on rocket landing tasks. The project emphasizes safe landings and fuel efficiency while demonstrating convergence, good ML practices, and project cohesion.

## What it Does

This project implements and compares multiple reinforcement learning algorithms on Gymnasium's LunarLander-v3 environment, which simulates rocket landing with continuous state space and discrete actions. While earlier coursework involved basic tabular Q-learning on the simple Taxi environment (discrete states) and a basic DQN implementation on LunarLander (single network, minimal hyperparameter exploration), our project significantly extends this foundation with multi-network baselines (DQN, A2C), optimizer comparisons (Adam vs RMSprop), reward shaping through custom reward functions, extensive training (>50k training episodes), GPU-accelerated training with automatic device detection, and rigorous evaluation on separate validation/test seed sets. Unlike the coursework's simpler DQN with a single network architecture and minimal hyperparameter tuning, this project implements advanced deep RL techniques including experience replay buffers, target networks with periodic updates, separate policy and value networks for A2C, comprehensive regularization (L2 weight decay, dropout, early stopping, gradient clipping), hyperparameter sweep frameworks, and proper train/validation/test splits using seed-based evaluation. The project trains Deep Q-Network (DQN) agents with experience replay and target networks, as well as Actor-Critic (A2C) agents with separate policy and value networks. Both algorithms are evaluated on their ability to achieve safe landings while minimizing fuel consumption, with comprehensive metrics tracking success rates, crash rates, fuel usage, and episode returns. The project includes custom reward functions, train/validation/test splits, regularization techniques, hyperparameter tuning, and visual demonstrations of agent behavior.

### Why Rocket Landing?

Rocket landing represents a critical real-world challenge in aerospace engineering and autonomous systems. Successfully landing a rocket requires precise control under uncertainty, balancing multiple competing objectives: achieving a safe landing within the designated landing pad, minimizing fuel consumption for cost efficiency, and maintaining smooth control to avoid structural damage. This problem is particularly relevant given recent advances in reusable rocket technology (e.g., SpaceX Falcon 9, Blue Origin New Shepard), where autonomous landing systems are essential for economic viability. The LunarLander-v3 environment captures the core challenges of this problem: continuous state space (position, velocity, angle, angular velocity), discrete control actions (engine firings), and stochastic dynamics. By solving this problem with reinforcement learning, we demonstrate how modern ML techniques can address complex control tasks with safety and efficiency constraints.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Note**: LunarLander-v3 requires Box2D. If installation fails, see [docs/INSTALL_BOX2D.md](docs/INSTALL_BOX2D.md) for troubleshooting.

### Run All Experiments

Train all agents and generate comparison plots:

```bash
python src/scripts/run_all_experiments.py
```

### Train Individual Agents

**DQN:**
```bash
# With Adam optimizer
python src/training/train_dqn.py --episodes 1000 --optimizer adam

# With RMSprop optimizer
python src/training/train_dqn.py --episodes 1000 --optimizer rmsprop
```

**A2C:**
```bash
# With Adam optimizer
python src/training/train_a2c.py --episodes 1000 --optimizer adam

# With SGD optimizer
python src/training/train_a2c.py --episodes 1000 --optimizer sgd
```

### Evaluate Trained Agents

```bash
# With reward wrapper (recommended - matches training setup)
python src/scripts/evaluate_agent.py --algorithm dqn --checkpoint models/dqn/dqn_adam_best.pt --num_episodes 50 --use_reward_wrapper

# Without reward wrapper (for comparison)
python src/scripts/evaluate_agent.py --algorithm dqn --checkpoint models/dqn/dqn_adam_best.pt --num_episodes 50
```

### Visual Testing

Test trained agents with visual rendering:

```bash
python src/scripts/test_visual.py --algorithm dqn --checkpoint models/dqn/dqn_best.pt --num_episodes 5
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

## Video Links

**Demo Video**: [Link to be added]
- Non-technical demonstration of the rocket landing agents
- Shows successful landings and key features
- Appropriate for general audience

**Technical Walkthrough**: [Link to be added]
- Code structure and architecture explanation
- ML techniques and key contributions
- Training process and evaluation methodology

## Evaluation

### Quantitative Results

The project tracks the following metrics on held-out test sets:

- **Success Rate**: Percentage of successful landings (landed safely within landing pad)
- **Crash Rate**: Percentage of crashes
- **Fuel Usage**: Average fuel consumption per episode
- **Episode Return**: Average cumulative reward
- **Episode Length**: Average episode duration

### Training Metrics

- **Learning Curves**: Episode returns over training progress, demonstrating convergence
- **Validation Performance**: Performance on validation seed sets for model selection
- **Optimizer Comparison**: Adam vs RMSprop for DQN, showing convergence speed differences

### Results Location

After training:
- **Plots**: Saved to `data/plots/` (learning curves, comparison plots, metrics)
- **Logs**: Saved to `data/logs/` (episode-by-episode training logs in CSV format)
- **Models**: Saved to `models/` (best checkpoints and periodic checkpoints)

### Comprehensive Evaluation Report

For detailed quantitative analysis and comprehensive evaluation results, see **[docs/EVALUATION.md](docs/EVALUATION.md)**. This document includes:

- **Optimizer Comparison**: Detailed quantitative comparison of Adam vs RMSprop (DQN) and Adam vs SGD (A2C) with test set results, success rates, fuel efficiency, and performance metrics
- **Model Architecture Comparison**: Quantitative DQN vs A2C comparison showing success rates, returns, stability, and fuel efficiency trade-offs
- **Error Analysis**: Detailed failure mode analysis for all models including crash statistics, failure patterns, and visualizations
- **Ablation Study Results**: Quantitative analysis of design choices (experience replay, target networks, custom reward shaping) and their impact on performance
- **Summary and Conclusions**: Performance summary tables, key findings, and recommendations for real-world applications

All results are backed by quantitative data from `results/test_results.csv`, `results/validation_results.csv`, error analysis CSVs, and ablation study results.

### Key Findings

- DQN with experience replay and target network achieves stable learning with good sample efficiency
- Actor-Critic (A2C) provides policy gradient approach with value function estimation for stable updates
- Custom reward function balances safety and fuel efficiency objectives
- Optimizer choice (Adam vs RMSprop) affects convergence speed and final performance
- Early stopping based on validation performance prevents overfitting to training seeds

### Hyperparameter Tuning Framework

The project includes a systematic hyperparameter tuning framework (`src/hyperparameter_tuning/sweep.py`) that performs grid search over key hyperparameters including learning rates, discount factors, epsilon schedules, reward coefficients, entropy regularization, and network architectures. The framework uses validation-based selection to identify optimal configurations, logging results to CSV for analysis. This enables systematic exploration of the hyperparameter space and identification of configurations that generalize well across different environment seeds.

### Optimizer Comparison

The project compares multiple optimizers (Adam vs RMSprop) for DQN training, demonstrating how optimizer choice affects convergence speed and final performance. Adam typically provides faster initial convergence due to adaptive learning rates, while RMSprop can offer more stable training in some cases. The comparison is conducted on identical network architectures and hyperparameters, with results logged and visualized to show convergence differences. This analysis helps identify the most effective optimization strategy for the rocket landing task.

## Individual Contributions

This project was completed collaboratively by **Jay Parmar** and **Ryan Christ**. All code development was assisted by Cursor AI, with each component assigned as follows:

### Jay Parmar
- **DQN Agent** (`src/agents/dqn.py`): Experience replay buffer, target network, epsilon-greedy exploration
- **Policy Network** (`src/networks/policy_network.py`): Policy network architecture for A2C
- **Reward Wrapper** (`src/environments/reward_wrapper.py`): Custom reward function design and implementation
- **Trainer** (`src/training/trainer.py`): Core training loop, early stopping, validation logic
- **Train A2C** (`src/training/train_a2c.py`): A2C training script
- **Evaluator** (`src/evaluation/evaluator.py`): Evaluation metrics and statistics computation
- **Hyperparameter Sweep** (`src/hyperparameter_tuning/sweep.py`): Grid search framework
- **Config** (`src/utils/config.py`): Configuration management and parameter definitions
- **Device** (`src/utils/device.py`): GPU/CPU device detection and management
- **Run All Experiments** (`src/scripts/run_all_experiments.py`): Main experiment orchestration script
- **Test Visual** (`src/scripts/test_visual.py`): Visual testing and rendering functionality
- **Ablation Study** (`src/scripts/ablation_study.py`): Ablation study implementation

### Ryan Christ
- **A2C Agent** (`src/agents/a2c.py`): Actor-critic implementation with policy and value networks
- **DQN Network** (`src/networks/dqn_network.py`): Q-network architecture with dropout support
- **Value Network** (`src/networks/value_network.py`): Value function network for A2C
- **Train DQN** (`src/training/train_dqn.py`): DQN training script
- **Compare Agents** (`src/evaluation/compare_agents.py`): Agent comparison and visualization
- **Plotting** (`src/utils/plotting.py`): Learning curve generation and visualization utilities
- **Evaluate Agent** (`src/scripts/evaluate_agent.py`): Agent evaluation script
- **Generate Plots** (`src/scripts/generate_plots.py`): Plot generation from checkpoints
- **Error Analysis** (`src/scripts/error_analysis.py`): Error analysis and failure case investigation
- **Analyze Training** (`src/scripts/analyze_training.py`): Training log analysis and diagnostics

### Collaborative
- **Documentation**: README.md, SETUP.md, ATTRIBUTION.md, and other documentation files
- **Project Structure**: Overall architecture and module organization
- **Testing and Debugging**: Joint testing and refinement of all components
- **Hyperparameter Tuning**: Collaborative tuning and optimization of all agents

**Note**: All code development was assisted by Cursor AI. Both team members reviewed, tested, and refined all components regardless of primary authorship.

## Project Structure

```
cs372final/
├── src/                    # Source code
│   ├── agents/             # RL agents (DQN, A2C)
│   ├── environments/       # RL environments
│   ├── networks/           # Neural network architectures
│   ├── training/            # Training scripts and utilities
│   ├── evaluation/         # Evaluation utilities
│   ├── hyperparameter_tuning/ # Hyperparameter sweep framework
│   ├── utils/              # Utility modules
│   └── scripts/            # Main scripts
├── data/                   # Data files and outputs
│   ├── logs/               # Training logs (CSV files)
│   └── plots/              # Generated plots and figures
├── models/                 # Trained models and checkpoints
├── notebooks/              # Jupyter notebooks (if any)
├── videos/                 # Demo and technical walkthrough videos
├── docs/                   # Additional documentation
└── requirements.txt        # Python dependencies
```

## Design Choices and Justifications

This section explains the key design decisions made in the project and their rationale.

### Custom Reward Function

The standard LunarLander-v3 reward function provides sparse rewards primarily at episode termination (landing success or crash). Our custom reward function (`src/environments/reward_wrapper.py`) addresses this limitation by providing dense, shaped rewards that guide learning toward desired behaviors. The reward function includes:

- **Landing Bonus**: Large positive reward (+100) for successful landings, encouraging the agent to complete the task
- **Fuel Penalty**: Small negative reward per engine firing (-0.1 for main engine, -0.05 for side engines), encouraging fuel-efficient trajectories
- **Crash Penalty**: Large negative reward (-100) for crashes, strongly discouraging unsafe landings
- **Smoothness Penalty**: Small penalty for large action changes, encouraging stable control

These components are parameterized through `RewardConfig`, enabling hyperparameter tuning to balance safety vs. efficiency. The shaped rewards provide more learning signal than sparse rewards, leading to faster convergence and better final performance.

### Multiple RL Paradigms (DQN and A2C)

We implement both value-based (DQN) and policy-based (A2C) approaches to provide a comprehensive comparison of different RL paradigms. DQN uses Q-learning with function approximation, learning action values through temporal difference learning. A2C uses policy gradients with value function estimation, learning a policy directly through advantage-weighted updates. This dual approach allows us to:

1. Compare sample efficiency: DQN with experience replay can reuse past experiences, while A2C learns on-policy
2. Compare stability: DQN's target network provides stable targets, while A2C's value function reduces variance in policy gradients
3. Understand trade-offs: Value-based methods excel at discrete action spaces, while policy-based methods naturally handle stochastic policies

Both approaches are evaluated on the same task with the same metrics, enabling fair comparison of their strengths and weaknesses.

### Train/Validation/Test Split Using Seeds

Unlike supervised learning where data can be randomly split, RL environments are deterministic given a seed. We use seed-based splits to create distinct train/validation/test sets:

- **Training Seeds** (42-51): Used for agent training, with random seed selection per episode to ensure diversity
- **Validation Seeds** (100-109): Used for model selection and early stopping, evaluated periodically during training
- **Test Seeds** (200-209): Used for final evaluation only, ensuring unbiased performance estimates

This approach ensures that validation and test performance reflect true generalization to unseen environment configurations, not just memorization of training seeds. The seed-based split is implemented in `src/training/trainer.py` and `src/utils/config.py`, with validation performed every N episodes (configurable via `val_frequency`) using multiple episodes per seed (configurable via `val_episodes_per_seed`) to reduce variance.

### Regularization Techniques

To prevent overfitting and improve generalization, we employ multiple regularization techniques:

- **L2 Weight Decay**: Applied through optimizer configuration (default 1e-5), penalizing large weights to prevent overfitting to training seeds
- **Dropout**: Applied in Q-network and value network (configurable rate, default 0.0), randomly zeroing activations during training to prevent co-adaptation of neurons
- **Early Stopping**: Monitors validation performance and stops training when validation returns plateau, preventing overfitting to training data
- **Gradient Clipping**: Clips gradients to a maximum norm (default 0.5-1.0), preventing exploding gradients and improving training stability

These techniques work together to ensure the learned policies generalize well to unseen environment configurations, not just the specific seeds used during training.

## Additional Scripts

The project includes several utility scripts for analysis, visualization, and hyperparameter tuning:

### Hyperparameter Sweep Scripts

**`run_dqn_sweep_fast.py`**: Fast hyperparameter sweep for DQN agents
- Performs grid search over learning rates, discount factors, optimizers, and weight decay
- Supports fast mode for quicker sweeps (reduced episodes and validation overhead)
- Optional reduced grid for faster exploration (16 vs 72 combinations)
- Usage: `python src/scripts/run_dqn_sweep_fast.py --fast --reduced-grid --episodes 300`

**`run_a2c_sweep_fast.py`**: Fast hyperparameter sweep for A2C agents
- Performs grid search over learning rates, discount factors, entropy coefficients, value coefficients, and optimizers
- Supports fast mode and reduced grid options similar to DQN sweep
- Usage: `python src/scripts/run_a2c_sweep_fast.py --fast --reduced-grid --episodes 300`

### Analysis and Visualization Scripts

**`generate_plots.py`**: Regenerate learning curve plots from saved checkpoints
- Loads trained agents from checkpoint files and generates learning curves
- Useful for regenerating plots without retraining
- Usage: `python src/scripts/generate_plots.py`

**`analyze_training.py`**: Analyze training logs and generate statistics
- Reads CSV training logs and computes statistics (mean, std, min, max returns)
- Generates learning curves with moving averages
- Provides detailed analysis of training progress
- Usage: `python src/scripts/analyze_training.py --log data/logs/dqn_training_log.csv`

**`error_analysis.py`**: Analyze failure cases and agent behavior
- Identifies when and why agents fail to land successfully
- Analyzes crash patterns, fuel usage in failure cases, and common failure modes
- Generates visualizations of failure statistics
- Usage: `python src/scripts/error_analysis.py --algorithm dqn --checkpoint models/dqn/dqn_best.pt`

**`ablation_study.py`**: Ablation study comparing design choices
- Compares DQN with and without key components (experience replay, target network, reward shaping)
- Demonstrates the impact of each component on performance
- Generates comparison plots and saves results to CSV
- Usage: `python src/scripts/ablation_study.py`

## Attribution

See [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for credits and external resources used.

## License

This project is for educational purposes (CS372 final project).
