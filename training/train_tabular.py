"""
Training script for tabular Q-learning on toy rocket environment.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tabular_q_learning import TabularQLearning
from environments.toy_rocket import ToyRocketEnv
from training.trainer import train_agent, create_env_factory, set_seed
from utils.config import TrainingConfig
from utils.plotting import plot_learning_curve


def main():
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints/tabular_q", help="Checkpoint directory")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Plot directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = ToyRocketEnv()
    state_space_size = env.get_state_space_size()
    action_space_size = env.action_space.n
    
    # Create agent
    agent = TabularQLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210))
    )
    
    # Create environment factory
    env_factory = create_env_factory("ToyRocket")
    
    # Train agent
    train_stats = train_agent(
        agent,
        env_factory,
        config,
        agent_name="tabular_q",
        algorithm_name="Tabular Q-Learning",
        environment_name="ToyRocket",
        save_dir=args.save_dir,
        log_dir="logs",
        plot_dir=args.plot_dir
    )
    
    # Save agent
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(os.path.join(args.save_dir, "tabular_q_final.pkl"))
    
    # Plot learning curve
    plot_learning_curve(
        agent.episode_returns,
        title="Tabular Q-Learning Learning Curve",
        save_path=os.path.join(args.plot_dir, "tabular_q_learning_curve.png")
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation return: {max(train_stats['val_returns']) if train_stats['val_returns'] else 'N/A'}")
    print(f"Final training return: {agent.episode_returns[-1] if agent.episode_returns else 'N/A'}")


if __name__ == "__main__":
    main()

