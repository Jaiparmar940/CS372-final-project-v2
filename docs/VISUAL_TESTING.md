# Visual Testing Guide

This guide explains how to test your trained RL agents visually in the LunarLander environment.

## Prerequisites

The LunarLander environment requires Box2D. Install it with:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

**Note**: The project uses LunarLander-v3, which requires Box2D dependencies.

## Quick Start

### 1. Train an Agent First

Before visual testing, you need a trained agent:

```bash
# Train DQN
python training/train_dqn.py --episodes 1000 --optimizer adam


# Or train A2C
python training/train_a2c.py --episodes 1000 --optimizer adam
```

### 2. Test Visually

Once you have a checkpoint, test it visually:

```bash
# Basic visual test
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5

# Slower rendering (easier to see)
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5 --delay 0.05

# Test without reward wrapper
python scripts/test_visual.py --algorithm dqn --checkpoint checkpoints/dqn/dqn_best.pt --num_episodes 5 --no_wrapper
```

## Options

- `--algorithm`: Algorithm name (`dqn` or `a2c`)
- `--checkpoint`: Path to checkpoint file (usually `*_best.pt`)
- `--num_episodes`: Number of episodes to run (default: 5)
- `--render_mode`: Rendering mode (`human`, `rgb_array`, or `none`)
- `--delay`: Delay between frames in seconds (default: 0.01)
- `--no_wrapper`: Don't use reward wrapper

## What You'll See

The script will:
1. Load your trained agent
2. Open a window showing the LunarLander environment
3. Run episodes with the agent controlling the rocket
4. Display real-time statistics:
   - Episode return
   - Number of steps
   - Outcome (successful landing, crash, or timeout)
   - Fuel usage (if tracked)
5. Show a summary at the end with success/crash rates

## Example Output

```
================================================================================
VISUAL TESTING - LunarLander Environment
================================================================================
Algorithm: dqn
Checkpoint: checkpoints/dqn/dqn_best.pt
Render mode: human
Number of episodes: 5

Agent loaded successfully from checkpoints/dqn/dqn_best.pt

================================================================================
Episode 1/5
================================================================================
  Return: 245.32
  Steps: 234
  Outcome: ✓ SUCCESSFUL LANDING
  Fuel used: 12.5

...

================================================================================
SUMMARY
================================================================================
Mean return: 238.45
Std return: 15.23
Mean episode length: 245.3
Success rate: 4/5 (80.0%)
Crash rate: 1/5 (20.0%)
================================================================================
```

## Troubleshooting

### "Box2D is not installed"
Install Box2D dependencies:
```bash
pip install swig
pip install "gymnasium[box2d]"
```

### "Box2D is not installed"
Install Box2D dependencies:
```bash
pip install swig
pip install "gymnasium[box2d]"
```
Or install all requirements:
```bash
pip install -r requirements.txt
```

### Window doesn't open
- Make sure you're using `--render_mode human` (default)
- Check that you have a display available (not running on headless server)
- Try `--render_mode rgb_array` to capture frames without display

### Agent performs poorly
- Make sure you're using the best checkpoint (`*_best.pt`)
- Check that the agent was trained for enough episodes
- Verify the checkpoint matches the algorithm type

## Tips

- Use `--delay 0.05` or higher to slow down rendering for easier observation
- Test multiple episodes to see consistency
- Compare different algorithms using the same checkpoint structure
- Use `--no_wrapper` to test with the base environment rewards

