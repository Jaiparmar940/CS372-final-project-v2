# Rocket Lander RL Project

A comprehensive reinforcement learning project comparing different RL paradigms on a rocket landing task. The project emphasizes safe landings and fuel efficiency while demonstrating convergence, good ML practices, and project cohesion.

## Project Overview

This project implements and compares multiple reinforcement learning algorithms on rocket landing tasks:

- **Tabular Q-Learning**: Basic Q-learning on a simple discrete toy environment
- **Deep Q-Network (DQN)**: Value-based deep RL with experience replay and target networks
- **REINFORCE**: Policy gradient method using Monte Carlo returns
- **Actor-Critic (A2C)**: Advantage Actor-Critic with separate policy and value networks

All deep RL agents are trained on Gymnasium's LunarLander-v3 environment, which simulates rocket landing with continuous state space and discrete actions.

## Why Rocket Landing?

Rocket landing is a challenging real-world problem that requires:
- **Safety**: Successful landings without crashes
- **Fuel Efficiency**: Minimizing fuel consumption
- **Smooth Control**: Stable, controlled descent

These objectives often conflict, making it an ideal testbed for comparing different RL approaches and reward designs.

## Quick Start

### Installation

See [SETUP.md](SETUP.md) for detailed installation instructions.

**Quick install:**
```bash
pip install -r requirements.txt
```

**Note**: LunarLander-v3 requires Box2D. If installation fails, see [INSTALL_BOX2D.md](INSTALL_BOX2D.md) for troubleshooting.

### Training Agents

**Note**: All scripts should be run from the project root directory. The scripts automatically handle Python path setup.

#### Tabular Q-Learning (Toy Environment)

```bash
python training/train_tabular.py --episodes 1000
```

#### DQN

```bash
# With Adam optimizer
python training/train_dqn.py --episodes 1000 --optimizer adam

# With RMSprop optimizer
python training/train_dqn.py --episodes 1000 --optimizer rmsprop
```

#### REINFORCE

```bash
python training/train_reinforce.py --episodes 1000 --optimizer adam
```

#### A2C

```bash
python training/train_a2c.py --episodes 1000 --optimizer adam
```

### Run All Experiments

```bash
python scripts/run_all_experiments.py
```

This will train all agents and generate comparison plots.

### Evaluation

After training, evaluate agents on test set:

**Using the evaluation script:**
```bash
python scripts/evaluate_agent.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 50
```

**Or programmatically:**
```python
from evaluation.compare_agents import compare_all_agents
from training.trainer import create_env_factory

# Load trained agents and compare
results = compare_all_agents(
    agents,
    env_factory,
    val_seeds,
    test_seeds
)
```

### Training Diagnostics

The project includes comprehensive diagnostics for monitoring training health:

1. **CONFIG SUMMARY**: Printed at start of each training run with all hyperparameters
2. **CSV Logging**: Every episode logged to `logs/{agent_name}_training_log.csv`
3. **Last 50 Episodes**: Formatted table displayed at end of training
4. **Analysis Script**: Analyze logs and generate learning curves
5. **Evaluation Script**: Evaluate trained agents on test seeds

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for detailed usage.

### Visual Testing

Test trained agents visually in the LunarLander environment:

```bash
# Test DQN agent with visual rendering
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5

# Test with slower rendering (easier to see)
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5 --delay 0.05

# Test without reward wrapper
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5 --no_wrapper
```

This will open a window showing the rocket landing attempts in real-time.

## Project Structure

```
cs372final/
├── environments/          # RL environments
│   ├── toy_rocket.py      # Simple discrete tabular environment
│   └── reward_wrapper.py  # Custom reward function wrapper
├── agents/                # RL agents
│   ├── tabular_q_learning.py
│   ├── dqn.py
│   ├── reinforce.py
│   └── a2c.py
├── networks/              # Neural network architectures
│   ├── dqn_network.py
│   ├── policy_network.py
│   └── value_network.py
├── training/              # Training scripts and utilities
│   ├── trainer.py
│   ├── train_tabular.py
│   ├── train_dqn.py
│   ├── train_reinforce.py
│   └── train_a2c.py
├── evaluation/           # Evaluation utilities
│   ├── evaluator.py
│   └── compare_agents.py
├── hyperparameter_tuning/ # Hyperparameter sweep framework
│   └── sweep.py
├── utils/                 # Utility modules
│   ├── device.py
│   ├── config.py
│   └── plotting.py
└── scripts/               # Main scripts
    ├── run_all_experiments.py
    └── generate_plots.py
```

## Key Features

### 1. Tabular Q-Learning
- **Location**: `agents/tabular_q_learning.py`
- Epsilon-greedy exploration with decay
- Q-table updates using Bellman equation
- Clear convergence on toy environment

### 2. DQN Components
- **Replay Buffer**: `agents/dqn.py` (ExperienceReplay class)
- **Target Network**: `agents/dqn.py` (periodic updates)
- **Custom Architecture**: `networks/dqn_network.py`
- Experience replay, target network, gradient clipping

### 3. Policy Methods
- **REINFORCE**: `agents/reinforce.py` - Monte Carlo policy gradient
- **A2C**: `agents/a2c.py` - Separate policy and value networks

### 4. Custom Reward Function
- **Location**: `environments/reward_wrapper.py`
- Parameterized reward with coefficients for:
  - Landing success bonus
  - Fuel consumption penalty
  - Crash penalty
  - Smoothness penalty

### 5. Train/Validation/Test Split
- **Location**: `training/trainer.py`
- Different random seeds for train/val/test sets
- Early stopping based on validation performance

### 6. Regularization Techniques
- **L2 Weight Decay**: Configurable in optimizer config
- **Dropout**: `networks/dqn_network.py` and `networks/value_network.py`
- **Early Stopping**: `training/trainer.py` (EarlyStopping class)
- **Gradient Clipping**: All agent training loops

### 7. Hyperparameter Tuning
- **Location**: `hyperparameter_tuning/sweep.py`
- Grid search framework
- CSV logging of results

### 8. Optimizer Comparison
- Adam vs RMSprop for DQN
- Adam vs SGD for policy methods
- Configurable in training scripts

### 9. Learning Rate Scheduling
- Step decay and ReduceLROnPlateau
- Configurable in optimizer config

### 10. GPU Support
- **Location**: `utils/device.py`
- Automatic CUDA detection
- Falls back to CPU if GPU unavailable

## Results and Plots

After training, plots are saved to `plots/`:
- Learning curves for each agent
- Comparison plots across algorithms
- Metrics comparison (success rate, fuel usage, etc.)

Results are saved to `results/`:
- Validation and test metrics in CSV format

## Evaluation Metrics

The project tracks metrics directly related to rocket landing:
- **Success Rate**: Percentage of successful landings
- **Crash Rate**: Percentage of crashes
- **Fuel Usage**: Average fuel consumption per episode
- **Episode Return**: Average cumulative reward
- **Episode Length**: Average episode duration

## Key Findings

(To be filled in after running experiments)

- Tabular Q-learning shows clear convergence on toy environment
- DQN with experience replay and target network achieves stable learning
- Policy gradient methods (REINFORCE, A2C) provide alternative approach
- Custom reward function balances safety and fuel efficiency
- Optimizer choice (Adam vs RMSprop) affects convergence speed

## Videos

**Demo Video**: [Link to be added]
- Non-technical demonstration of the rocket landing agents
- Shows successful landings and key features
- Appropriate for general audience

**Technical Walkthrough**: [Link to be added]
- Code structure and architecture explanation
- ML techniques and key contributions
- Training process and evaluation methodology

## Individual Contributions

This project was completed individually. All components including environment design, agent implementations, training infrastructure, evaluation metrics, and documentation were developed as part of this final project.

## Design Choices and Justifications

### Custom Reward Function
The custom reward wrapper (`environments/reward_wrapper.py`) was designed to explicitly balance three competing objectives:
1. **Safety**: Large bonus for successful landings, large penalty for crashes
2. **Fuel Efficiency**: Penalty proportional to engine usage
3. **Smooth Control**: Penalty for large action changes

This parameterized design allows systematic exploration of reward shaping effects, which is critical for understanding how different RL algorithms respond to reward design.

### Multiple RL Paradigms
We compare four different RL approaches to understand their relative strengths:
- **Tabular Q-Learning**: Baseline for understanding basic RL concepts on a simple environment
- **DQN**: Value-based deep RL with experience replay for sample efficiency
- **REINFORCE**: Policy gradient method that directly optimizes policy
- **A2C**: Actor-critic combines policy gradients with value function estimation

This comparison demonstrates how different algorithmic choices affect learning dynamics and final performance.

### Train/Validation/Test Split via Seeds
RL environments are deterministic given a seed. We use different seed ranges for train/val/test sets to ensure:
- Training on diverse initial conditions
- Validation for hyperparameter tuning and early stopping
- Test set for unbiased final evaluation

This approach provides proper evaluation methodology while maintaining environment determinism.

### Regularization Techniques
We apply multiple regularization techniques to prevent overfitting:
- **L2 Weight Decay**: Prevents weight explosion
- **Dropout**: Reduces overfitting in value networks
- **Early Stopping**: Prevents overfitting to training seeds
- **Gradient Clipping**: Prevents exploding gradients during training

These techniques are essential for stable deep RL training.

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for credits and external resources used.

## License

This project is for educational purposes (CS372 final project).

