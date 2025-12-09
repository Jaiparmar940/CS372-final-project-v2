# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create a training script for A2C agent with command-line arguments for hyperparameters"

"""
Training script for A2C (Advantage Actor-Critic) agent.
"""

import argparse
import os
import sys

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
from agents.a2c import A2CAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, set_seed
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train A2C agent")
    
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate (slightly higher for faster improvement)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy_coef", type=float, default=0.005, help="Entropy regularization coefficient (lower to reduce exploration once")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of steps before update (shorter for faster updates, longer for better")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop", "sgd"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay (small amount for regularization)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for value network")
    parser.add_argument("--gradient_clip", type=float, default=0.5, help="Gradient clipping threshold (tighter for stability)")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size (smaller = faster, default: 128)")
    parser.add_argument("--val_frequency", type=int, default=30, help="Validation frequency in episodes (higher = faster, default: 20)")
    parser.add_argument("--val_episodes_per_seed", type=int, default=1, help="Validation episodes per seed (1 = fastest, default: 1)")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler (disabled by default to avoid over-decay)")
    parser.add_argument("--no_scheduler", action="store_false", dest="use_scheduler", help="Disable learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="models/a2c", help="Checkpoint directory")
    parser.add_argument("--plot_dir", type=str, default="data/plots", help="Plot directory")
    parser.add_argument("--agent_name", type=str, default=None, help="Custom agent name for logging (default: 'a2c' or 'a2c_sgd')")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine agent name
    if args.agent_name is None:
        agent_name = "a2c" if args.optimizer == "adam" else f"a2c_{args.optimizer}"
    else:
        agent_name = args.agent_name
    
    # Create environment factory
    reward_config = RewardConfig()
    
    def lunar_lander_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    # Create agent
    device = get_device()
    
    network_config = NetworkConfig(
        hidden_sizes=[args.hidden_size, args.hidden_size],
        dropout_rate=args.dropout,
        use_dropout=args.dropout > 0.0
    )
    
    optimizer_config = OptimizerConfig(
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler
    )
    
    agent = A2CAgent(
        state_dim=8,
        action_dim=4,
        network_config=network_config,
        optimizer_config=optimizer_config,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        n_steps=args.n_steps,
        gradient_clip=args.gradient_clip,
        device=device
    )
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210)),
        val_frequency=args.val_frequency,
        val_episodes_per_seed=args.val_episodes_per_seed
    )
    
    # Train agent
    env_name_str = "LunarLander-v3 (with RocketRewardWrapper)"
    train_stats = train_agent(
        agent,
        lunar_lander_factory,
        config,
        agent_name=agent_name,
        algorithm_name="A2C",
        environment_name=env_name_str,
        reward_config=reward_config,
        save_dir=args.save_dir,
        log_dir="data/logs",
        plot_dir=args.plot_dir
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation return: {max(train_stats['val_returns']) if train_stats['val_returns'] else 'N/A'}")
    print(f"Final training return: {agent.episode_returns[-1] if agent.episode_returns else 'N/A'}")


if __name__ == "__main__":
    main()
