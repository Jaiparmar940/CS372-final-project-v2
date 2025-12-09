# This file was created by Cursor and fine tuned by Jaivir Parmar and Ryan Christ.
# To recreate this file, prompt Cursor with: "Create a training script for DQN agent on LunarLander-v3 with command-line arguments for hyperparameters and configuration"

"""
Training script for DQN agent on LunarLander-v3.
"""

import argparse
import os
import sys
import gymnasium as gym

# Add project root to path
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from agents.dqn import DQNAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, create_env_factory, set_seed
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--target_update_freq", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop", "sgd"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--gradient_clip", type=float, default=10.0, help="Gradient clipping threshold")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="models/dqn", help="Checkpoint directory")
    parser.add_argument("--plot_dir", type=str, default="data/plots", help="Plot directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment to get dimensions
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Network configuration
    network_config = NetworkConfig(
        hidden_sizes=[128, 128],
        dropout_rate=args.dropout,
        use_dropout=args.dropout > 0.0
    )
    
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
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        network_config=network_config,
        optimizer_config=optimizer_config,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_frequency=args.target_update_freq,
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
        agent_name="dqn",
        algorithm_name="DQN",
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

