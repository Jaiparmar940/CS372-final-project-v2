"""
Run all training experiments for comparison.
Trains all agents and generates comparison plots.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from agents.tabular_q_learning import TabularQLearning
from agents.dqn import DQNAgent
from agents.reinforce import REINFORCEAgent
from agents.a2c import A2CAgent
from environments.toy_rocket import ToyRocketEnv
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, create_env_factory, set_seed, evaluate_on_seeds
from evaluation.compare_agents import compare_all_agents, plot_learning_curves_comparison
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device


def main():
    print("="*80)
    print("Rocket Lander RL - Running All Experiments")
    print("="*80)
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=500,  # Reduced for faster demo
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210))
    )
    
    # Reward configuration
    reward_config = RewardConfig()
    
    # Create environment factories
    def toy_env_factory(seed):
        env = ToyRocketEnv()
        set_seed(seed)
        return env
    
    def lunar_lander_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    agents = {}
    
    # 1. Train Tabular Q-Learning on toy environment
    print("\n" + "="*80)
    print("1. Training Tabular Q-Learning on Toy Rocket Environment")
    print("="*80)
    
    toy_env = ToyRocketEnv()
    state_space_size = toy_env.get_state_space_size()
    action_space_size = toy_env.action_space.n
    
    tabular_agent = TabularQLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.99
    )
    
    train_stats_tabular = train_agent(
        tabular_agent,
        toy_env_factory,
        config,
        agent_name="tabular_q",
        save_dir="checkpoints/tabular_q",
        plot_dir="plots"
    )
    agents["Tabular Q-Learning"] = tabular_agent
    
    # 2. Train DQN with Adam
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
    
    # 3. Train DQN with RMSprop
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
    
    # 4. Train REINFORCE with Adam
    print("\n" + "="*80)
    print("4. Training REINFORCE with Adam Optimizer")
    print("="*80)
    
    reinforce_adam = REINFORCEAgent(
        state_dim=8,
        action_dim=4,
        network_config=NetworkConfig(hidden_sizes=[128, 128]),
        optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
        device=get_device()
    )
    
    train_stats_reinforce = train_agent(
        reinforce_adam,
        lunar_lander_factory,
        config,
        agent_name="reinforce",
        save_dir="checkpoints/reinforce",
        plot_dir="plots"
    )
    agents["REINFORCE (Adam)"] = reinforce_adam
    
    # 5. Train A2C with Adam
    print("\n" + "="*80)
    print("5. Training A2C with Adam Optimizer")
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
    
    # 6. Compare all agents
    print("\n" + "="*80)
    print("6. Comparing All Agents")
    print("="*80)
    
    # Only compare deep RL agents on LunarLander (tabular Q-learning is on different env)
    deep_rl_agents = {k: v for k, v in agents.items() if k != "Tabular Q-Learning"}
    
    comparison_results = compare_all_agents(
        deep_rl_agents,
        lunar_lander_factory,
        config.val_seeds,
        config.test_seeds,
        plot_dir="plots",
        results_dir="results"
    )
    
    # Plot learning curves comparison
    plot_learning_curves_comparison(deep_rl_agents, plot_dir="plots")
    
    print("\n" + "="*80)
    print("All experiments complete!")
    print("="*80)
    print("\nResults saved to:")
    print("  - Plots: plots/")
    print("  - Checkpoints: checkpoints/")
    print("  - Results: results/")


if __name__ == "__main__":
    main()

