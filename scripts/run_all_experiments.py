# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create a script that runs all training experiments for DQN and A2C agents and generates comparison plots"

"""
Run all training experiments for comparison.
Trains all agents and generates comparison plots.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, create_env_factory, set_seed, evaluate_on_seeds
from evaluation.compare_agents import compare_all_agents, plot_learning_curves_comparison
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device
from scripts.test_visual import test_visual


def main():
    parser = argparse.ArgumentParser(description="Run all training experiments for comparison")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes (default: 500)")
    args = parser.parse_args()
    
    print("="*80)
    print("Rocket Lander RL - Running All Experiments")
    print("="*80)
    print(f"Training with {args.episodes} episodes per agent")
    print("="*80)
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210))
    )
    
    # Reward configuration
    reward_config = RewardConfig()
    
    # Create environment factories
    def lunar_lander_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    agents = {}
    
    # 1. Train DQN with Adam
    print("\n" + "="*80)
    print("2. Training DQN with Adam Optimizer")
    print("="*80)
    
    dqn_adam = DQNAgent(
        state_dim=8,
        action_dim=4,
        network_config=NetworkConfig(hidden_sizes=[128, 128]),
        optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
        device=get_device()
    )
    
    train_stats_dqn_adam = train_agent(
        dqn_adam,
        lunar_lander_factory,
        config,
        agent_name="dqn_adam",
        save_dir="checkpoints/dqn",
        plot_dir="plots"
    )
    agents["DQN (Adam)"] = dqn_adam
    
    # 2. Train DQN with RMSprop
    print("\n" + "="*80)
    print("3. Training DQN with RMSprop Optimizer")
    print("="*80)
    
    dqn_rmsprop = DQNAgent(
        state_dim=8,
        action_dim=4,
        network_config=NetworkConfig(hidden_sizes=[128, 128]),
        optimizer_config=OptimizerConfig(optimizer="rmsprop", learning_rate=1e-3),
        device=get_device()
    )
    
    train_stats_dqn_rmsprop = train_agent(
        dqn_rmsprop,
        lunar_lander_factory,
        config,
        agent_name="dqn_rmsprop",
        save_dir="checkpoints/dqn",
        plot_dir="plots"
    )
    agents["DQN (RMSprop)"] = dqn_rmsprop
    
    # 3. Train A2C with Adam
    print("\n" + "="*80)
    print("3. Training A2C with Adam Optimizer")
    print("="*80)
    
    a2c_adam = A2CAgent(
        state_dim=8,
        action_dim=4,
        network_config=NetworkConfig(hidden_sizes=[128, 128]),
        optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
        device=get_device()
    )
    
    train_stats_a2c = train_agent(
        a2c_adam,
        lunar_lander_factory,
        config,
        agent_name="a2c",
        save_dir="checkpoints/a2c",
        plot_dir="plots"
    )
    agents["A2C (Adam)"] = a2c_adam
    
    # 4. Compare all agents
    print("\n" + "="*80)
    print("4. Comparing All Agents")
    print("="*80)
    
    comparison_results = compare_all_agents(
        agents,
        lunar_lander_factory,
        config.val_seeds,
        config.test_seeds,
        plot_dir="plots",
        results_dir="results"
    )
    
    # Plot learning curves comparison
    plot_learning_curves_comparison(agents, plot_dir="plots")
    
    # 5. Visual demonstrations for each algorithm
    print("\n" + "="*80)
    print("5. Visual Demonstrations")
    print("="*80)
    
    # Map agent names to their checkpoint paths and algorithm types
    agent_demos = [
        ("DQN (Adam)", "dqn", "checkpoints/dqn/dqn_adam_best.pt"),
        ("DQN (RMSprop)", "dqn", "checkpoints/dqn/dqn_rmsprop_best.pt"),
        ("A2C (Adam)", "a2c", "checkpoints/a2c/a2c_best.pt"),
    ]
    
    for agent_name, algorithm, checkpoint_path in agent_demos:
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*80}")
            print(f"Visual Demonstration: {agent_name}")
            print(f"{'='*80}")
            try:
                test_visual(
                    algorithm=algorithm,
                    checkpoint_path=checkpoint_path,
                    num_episodes=3,  # Run 3 episodes for demonstration
                    use_reward_wrapper=True,
                    render_mode="human",
                    delay=0.01
                )
            except Exception as e:
                print(f"Error running visual demonstration for {agent_name}: {e}")
                print("Skipping visual demo (this is okay if display is not available)")
        else:
            print(f"\nWarning: Checkpoint not found for {agent_name}: {checkpoint_path}")
            print("Skipping visual demonstration")
    
    print("\n" + "="*80)
    print("All experiments complete!")
    print("="*80)
    print("\nResults saved to:")
    print("  - Plots: plots/")
    print("  - Checkpoints: checkpoints/")
    print("  - Results: results/")


if __name__ == "__main__":
    main()

