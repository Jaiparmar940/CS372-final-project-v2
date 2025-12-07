# Training Diagnostics Guide

This document explains how to use the diagnostic features added to the RL training project.

## Features

### 1. CONFIG SUMMARY

At the start of each training run, a clear CONFIG SUMMARY block is printed showing:
- Algorithm name
- Environment name
- All key hyperparameters
- Reward configuration (if using wrapper)

**Example:**
```
================================================================================
CONFIG SUMMARY
================================================================================
Algorithm: DQN
Environment: LunarLander-v2 (with RocketRewardWrapper)
Agent Name: dqn

Training Configuration:
  num_episodes: 1000
  max_steps_per_episode: 1000
  gamma (discount factor): 0.99
  ...
```

### 2. Per-Episode CSV Logging

Every episode is logged to a CSV file in `logs/` directory with:
- episode index
- episode return
- episode length
- current epsilon (if applicable)
- current learning rate (if using scheduler)
- validation return (when evaluated)

**File location:** `logs/{agent_name}_training_log.csv`

### 3. Console Output

The last 50 episodes are clearly displayed in a formatted table at the end of training:
```
================================================================================
LAST 50 EPISODES SUMMARY
================================================================================
Episode    Return       Length     Epsilon      LR           Val Return  
--------------------------------------------------------------------------------
...
```

### 4. Analysis Script

Analyze training logs and generate learning curves:

```bash
python scripts/analyze_training.py --log logs/tabular_q_training_log.csv --window 100
```

Options:
- `--log`: Path to CSV training log
- `--window`: Window size for moving average (default: 100)
- `--save_plot`: Path to save plot (optional)

**Output includes:**
- Return statistics (mean, std, min, max, first/last 10 episodes)
- Episode length statistics
- Epsilon and learning rate statistics
- Validation statistics (if available)
- Learning curve plot with moving average

### 5. Evaluation Script

Evaluate a trained agent on test seeds:

```bash
python scripts/evaluate_agent.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 50
```

Options:
- `--algorithm`: Algorithm name (tabular_q, dqn, reinforce, a2c)
- `--checkpoint`: Path to checkpoint file
- `--num_episodes`: Number of episodes to evaluate (default: 50)
- `--test_seeds`: Custom test seeds (default: 200-209)
- `--use_reward_wrapper`: Use reward wrapper for LunarLander

**Output includes:**
- Mean return and standard deviation
- Success rate (% successful landings)
- Crash rate
- Mean fuel usage (if tracked)
- Episode length statistics

## Usage Examples

### Training with Diagnostics

```bash
# Train tabular Q-learning
python training/train_tabular.py --episodes 1000

# Train DQN
python training/train_dqn.py --episodes 1000 --optimizer adam

# Train REINFORCE
python training/train_reinforce.py --episodes 1000 --optimizer adam

# Train A2C
python training/train_a2c.py --episodes 1000 --optimizer adam
```

All training runs will:
1. Print CONFIG SUMMARY at start
2. Log every episode to CSV
3. Display last 50 episodes at end
4. Save checkpoints

### Analyzing Results

```bash
# Analyze training log
python scripts/analyze_training.py --log logs/tabular_q_training_log.csv --window 100 --save_plot plots/analysis.png

# Evaluate trained agent
python scripts/evaluate_agent.py --algorithm tabular_q --checkpoint checkpoints/tabular_q/tabular_q_best.pkl --num_episodes 50
```

## Pasting to ChatGPT

The outputs are designed to be easily pasted into ChatGPT for debugging:

1. **CONFIG SUMMARY**: Copy the entire block from training start
2. **Episode Logs**: Copy the "LAST 50 EPISODES SUMMARY" table
3. **Evaluation Summary**: Copy the evaluation output

All outputs use clear formatting with separators (===) for easy identification.

## File Locations

- **Training logs**: `logs/{agent_name}_training_log.csv`
- **Checkpoints**: `checkpoints/{agent_name}/`
- **Plots**: `plots/`
- **Analysis plots**: Specify with `--save_plot` flag

