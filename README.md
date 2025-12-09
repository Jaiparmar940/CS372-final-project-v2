# Rocket Lander RL Project

A comprehensive reinforcement learning project comparing Deep Q-Network (DQN) and Actor-Critic (A2C) algorithms on rocket landing tasks. The project emphasizes safe landings and fuel efficiency while demonstrating convergence, good ML practices, and project cohesion.

## What it Does

This project implements and compares multiple reinforcement learning algorithms on Gymnasium's LunarLander-v3 environment, which simulates rocket landing with continuous state space and discrete actions. While earlier coursework involved basic tabular Q-learning on the simple Taxi environment (discrete states) and a basic DQN implementation on LunarLander (single network, minimal hyperparameter exploration), our project significantly extends this foundation with multi-network baselines (DQN, A2C), optimizer comparisons (Adam vs RMSprop), reward shaping through custom reward functions, extensive training (>50k training episodes), GPU-accelerated training with automatic device detection, and rigorous evaluation on separate validation/test seed sets. Unlike the coursework's simpler DQN with a single network architecture and minimal hyperparameter tuning, this project implements advanced deep RL techniques including experience replay buffers, target networks with periodic updates, separate policy and value networks for A2C, comprehensive regularization (L2 weight decay, dropout, early stopping, gradient clipping), hyperparameter sweep frameworks, and proper train/validation/test splits using seed-based evaluation. The project trains Deep Q-Network (DQN) agents with experience replay and target networks, as well as Actor-Critic (A2C) agents with separate policy and value networks. Both algorithms are evaluated on their ability to achieve safe landings while minimizing fuel consumption, with comprehensive metrics tracking success rates, crash rates, fuel usage, and episode returns. The project includes custom reward functions, train/validation/test splits, regularization techniques, hyperparameter tuning, and visual demonstrations of agent behavior.

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
python src/training/train_a2c.py --episodes 1000 --optimizer adam
```

### Evaluate Trained Agents

```bash
python src/scripts/evaluate_agent.py --algorithm dqn --checkpoint models/dqn/dqn_best.pt --num_episodes 50
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

### Key Findings

- DQN with experience replay and target network achieves stable learning with good sample efficiency
- Actor-Critic (A2C) provides policy gradient approach with value function estimation for stable updates
- Custom reward function balances safety and fuel efficiency objectives
- Optimizer choice (Adam vs RMSprop) affects convergence speed and final performance
- Early stopping based on validation performance prevents overfitting to training seeds

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

## Attribution

See [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for credits and external resources used.

## License

This project is for educational purposes (CS372 final project).
