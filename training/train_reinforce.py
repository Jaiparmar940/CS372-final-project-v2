"""
Training script for REINFORCE agent on LunarLander-v3.
"""

import argparse
import os
import sys
import gymnasium as gym

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.reinforce import REINFORCEAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, set_seed
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop", "sgd"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--gradient_clip", type=float, default=10.0, help="Gradient clipping threshold")
    parser.add_argument("--use_baseline", action="store_true", help="Use baseline subtraction")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints/reinforce", help="Checkpoint directory")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Plot directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment to get dimensions
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Network configuration
    network_config = NetworkConfig(hidden_sizes=[128, 128])
    
    # Optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler
    )
    
    # Reward configuration
    reward_config = RewardConfig()
    
    # Create agent
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        network_config=network_config,
        optimizer_config=optimizer_config,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        use_baseline=args.use_baseline,
        gradient_clip=args.gradient_clip,
        device=get_device()
    )
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210))
    )
    
    # Create environment factory with reward wrapper
    def env_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    # Train agent
    env_name_str = "LunarLander-v3 (with RocketRewardWrapper)" if reward_config else "LunarLander-v3"
    train_stats = train_agent(
        agent,
        env_factory,
        config,
        agent_name="reinforce",
        algorithm_name="REINFORCE",
        environment_name=env_name_str,
        reward_config=reward_config,
        save_dir=args.save_dir,
        log_dir="logs",
        plot_dir=args.plot_dir
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation return: {max(train_stats['val_returns']) if train_stats['val_returns'] else 'N/A'}")
    print(f"Final training return: {agent.episode_returns[-1] if agent.episode_returns else 'N/A'}")


if __name__ == "__main__":
    main()

