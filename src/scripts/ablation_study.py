# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create an ablation study script that compares DQN with and without experience replay, target network, and reward shaping to demonstrate the impact of each component"

"""
Ablation study comparing different design choices in DQN.
Demonstrates the impact of key components: experience replay, target network, and reward shaping.
"""

import os
import sys
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
import numpy as np
from agents.dqn import DQNAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import train_agent, set_seed, evaluate_on_seeds
from utils.config import TrainingConfig, NetworkConfig, OptimizerConfig, RewardConfig
from utils.device import get_device
import pandas as pd
import matplotlib.pyplot as plt


def create_env_factory(reward_config=None):
    """Create environment factory."""
    def env_factory(seed):
        env = gym.make("LunarLander-v3")
        if reward_config:
            env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    return env_factory


def run_ablation_study():
    """Run ablation study comparing different design choices."""
    
    print("="*80)
    print("ABLATION STUDY: Impact of Design Choices")
    print("="*80)
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=500,  # Reduced for faster ablation study
        train_seeds=list(range(42, 47)),  # Smaller seed set for speed
        val_seeds=list(range(100, 105)),
        test_seeds=list(range(200, 205))
    )
    
    # Toggle blocks to re-use this script without deleting code.
    run_baseline = False
    run_no_replay = False
    run_no_target = False
    run_custom_reward = True  # Only run the last ablation
    
    results = []
    
    # Baseline: Full DQN with experience replay and target network
    print("\n" + "="*80)
    print("1. Baseline: Full DQN (with replay buffer and target network)")
    print("="*80)
    
    if run_baseline:
        baseline_agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(hidden_sizes=[128, 128]),
            optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
            replay_buffer_size=10000,
            batch_size=64,
            target_update_frequency=100,
            device=get_device()
        )
        
        env_factory = create_env_factory()
        train_stats_baseline = train_agent(
            baseline_agent,
            env_factory,
            config,
            agent_name="ablation_baseline",
            algorithm_name="DQN (Full)",
            environment_name="LunarLander-v3",
            save_dir="ablation_results/checkpoints",
            log_dir="ablation_results/logs",
            plot_dir="ablation_results/plots"
        )
        
        val_metrics_baseline = evaluate_on_seeds(
            baseline_agent, env_factory, config.val_seeds, num_episodes_per_seed=3
        )
        
        results.append({
            "configuration": "Baseline (Full DQN)",
            "replay_buffer": True,
            "target_network": True,
            "val_return": val_metrics_baseline["mean_return"],
            "val_std": val_metrics_baseline["std_return"],
            "final_train_return": train_stats_baseline["train_returns"][-1] if train_stats_baseline["train_returns"] else 0
        })
    
    # Ablation 1: DQN without experience replay (online learning only)
    print("\n" + "="*80)
    print("2. Ablation: DQN without Experience Replay (online learning only)")
    print("="*80)
    
    if run_no_replay:
        no_replay_agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(hidden_sizes=[128, 128]),
            optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
            replay_buffer_size=1,  # Effectively no replay
            batch_size=1,  # Online updates only
            target_update_frequency=100,
            device=get_device()
        )
        
        train_stats_no_replay = train_agent(
            no_replay_agent,
            env_factory,
            config,
            agent_name="ablation_no_replay",
            algorithm_name="DQN (No Replay)",
            environment_name="LunarLander-v3",
            save_dir="ablation_results/checkpoints",
            log_dir="ablation_results/logs",
            plot_dir="ablation_results/plots"
        )
        
        val_metrics_no_replay = evaluate_on_seeds(
            no_replay_agent, env_factory, config.val_seeds, num_episodes_per_seed=3
        )
        
        results.append({
            "configuration": "No Experience Replay",
            "replay_buffer": False,
            "target_network": True,
            "val_return": val_metrics_no_replay["mean_return"],
            "val_std": val_metrics_no_replay["std_return"],
            "final_train_return": train_stats_no_replay["train_returns"][-1] if train_stats_no_replay["train_returns"] else 0
        })
    
    # Ablation 2: DQN without target network (direct Q-learning)
    print("\n" + "="*80)
    print("3. Ablation: DQN without Target Network")
    print("="*80)
    
    if run_no_target:
        no_target_agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(hidden_sizes=[128, 128]),
            optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
            replay_buffer_size=10000,
            batch_size=64,
            target_update_frequency=1,  # Update every step (effectively no separate target)
            device=get_device()
        )
        
        train_stats_no_target = train_agent(
            no_target_agent,
            env_factory,
            config,
            agent_name="ablation_no_target",
            algorithm_name="DQN (No Target)",
            environment_name="LunarLander-v3",
            save_dir="ablation_results/checkpoints",
            log_dir="ablation_results/logs",
            plot_dir="ablation_results/plots"
        )
        
        val_metrics_no_target = evaluate_on_seeds(
            no_target_agent, env_factory, config.val_seeds, num_episodes_per_seed=3
        )
        
        results.append({
            "configuration": "No Target Network",
            "replay_buffer": True,
            "target_network": False,
            "val_return": val_metrics_no_target["mean_return"],
            "val_std": val_metrics_no_target["std_return"],
            "final_train_return": train_stats_no_target["train_returns"][-1] if train_stats_no_target["train_returns"] else 0
        })
    
    # Ablation 3: Different reward shaping (custom reward wrapper)
    print("\n" + "="*80)
    print("4. Ablation: Custom Reward Shaping")
    print("="*80)
    
    if run_custom_reward:
        reward_config = RewardConfig(
            landing_bonus=100.0,
            crash_penalty=-100.0,
            fuel_penalty=0.1,
            smoothness_penalty=0.05
        )
        
        custom_reward_env_factory = create_env_factory(reward_config)
        
        custom_reward_agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(hidden_sizes=[128, 128]),
            optimizer_config=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
            replay_buffer_size=10000,
            batch_size=64,
            target_update_frequency=100,
            device=get_device()
        )
        
        train_stats_custom = train_agent(
            custom_reward_agent,
            custom_reward_env_factory,
            config,
            agent_name="ablation_custom_reward",
            algorithm_name="DQN (Custom Reward)",
            environment_name="LunarLander-v3 (Custom Reward)",
            reward_config=reward_config,
            save_dir="ablation_results/checkpoints",
            log_dir="ablation_results/logs",
            plot_dir="ablation_results/plots"
        )
        
        val_metrics_custom = evaluate_on_seeds(
            custom_reward_agent, custom_reward_env_factory, config.val_seeds, num_episodes_per_seed=3
        )
        
        results.append({
            "configuration": "Custom Reward Shaping",
            "replay_buffer": True,
            "target_network": True,
            "val_return": val_metrics_custom["mean_return"],
            "val_std": val_metrics_custom["std_return"],
            "final_train_return": train_stats_custom["train_returns"][-1] if train_stats_custom["train_returns"] else 0
        })
    
    # Save results
    os.makedirs("ablation_results", exist_ok=True)
    if not results:
        print("No experiments were run. Skipping output.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_results/ablation_study_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # If fewer than 2 configurations, skip comparison plots
    if len(df) < 2:
        print("\nNot enough configurations to create comparison plots. Skipping plot generation.")
        return df
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation return comparison
    configs = df["configuration"].values
    val_returns = df["val_return"].values
    val_stds = df["val_std"].values
    
    ax1.bar(configs, val_returns, yerr=val_stds, capsize=5, alpha=0.7)
    ax1.set_ylabel("Validation Return")
    ax1.set_title("Impact of Design Choices on Validation Performance")
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Component comparison (requires baseline + others)
    components = []
    component_impact = []
    if len(val_returns) >= 2:
        # If baseline present at index 0, compute deltas
        baseline_val = val_returns[0]
        if len(val_returns) > 1:
            components.append("Replay Buffer / Variant")
            component_impact.append(baseline_val - val_returns[1])
        if len(val_returns) > 2:
            components.append("Target Network / Variant")
            component_impact.append(baseline_val - val_returns[2])
        if len(val_returns) > 3:
            components.append("Custom Reward vs Baseline")
            component_impact.append(val_returns[3] - baseline_val)
    
    # If we don't have enough points for component comparison, show empty chart
    colors = ['green' if x > 0 else 'red' for x in component_impact]
    ax2.barh(components, component_impact, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Impact on Validation Return")
    ax2.set_title("Individual Component Impact")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ablation_results/ablation_study_comparison.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: ablation_results/ablation_study_comparison.png")
    
    return df


if __name__ == "__main__":
    set_seed(42)
    results_df = run_ablation_study()
    print("\nAblation study complete! Results saved to ablation_results/")

