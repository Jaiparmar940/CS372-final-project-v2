# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create an evaluation script that loads trained agent checkpoints and evaluates them on held-out test seed sets"

"""
Evaluation entry point for trained agents.
Loads best checkpoint and evaluates on held-out test seeds.
"""

import argparse
import os
import sys

# Add project root to path
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
import torch
from training.trainer import set_seed, create_env_factory
from environments.reward_wrapper import RocketRewardWrapper
from utils.config import RewardConfig, TrainingConfig, OptimizerConfig
from evaluation.evaluator import compute_landing_metrics, print_evaluation_summary


def load_agent(algorithm: str, checkpoint_path: str):
    """
    Load agent from checkpoint.
    
    Args:
        algorithm: Algorithm name ("dqn", "a2c")
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded agent
    """
    from agents.dqn import DQNAgent
    from agents.a2c import A2CAgent
    from utils.device import get_device
    from utils.config import NetworkConfig, OptimizerConfig
    
    device = get_device()
    
    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        return agent, "LunarLander-v3"
    
    elif algorithm == "a2c":
        # Load checkpoint to get optimizer config first
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        optimizer_config = checkpoint.get("optimizer_config", OptimizerConfig())
        
        agent = A2CAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=optimizer_config,
            device=device
        )
        agent.load(checkpoint_path)
        return agent, "LunarLander-v3"
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["dqn", "a2c"],
                       help="Algorithm name")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=50,
                       help="Number of episodes to evaluate")
    parser.add_argument("--test_seeds", type=int, nargs="+", default=None,
                       help="Test seeds (default: 200-209)")
    parser.add_argument("--use_reward_wrapper", action="store_true",
                       help="Use reward wrapper for LunarLander")
    
    args = parser.parse_args()
    
    # Load agent
    print("="*80)
    print("EVALUATION")
    print("="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    try:
        agent, env_name = load_agent(args.algorithm, args.checkpoint)
        print(f"Agent loaded successfully from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return
    
    # Setup test seeds
    if args.test_seeds is None:
        test_seeds = list(range(200, 210))
    else:
        test_seeds = args.test_seeds
    
    print(f"Test seeds: {test_seeds}")
    print(f"Number of episodes: {args.num_episodes}")
    print()
    
    # Create environment factory
    reward_config = RewardConfig() if args.use_reward_wrapper else None
    def env_factory(seed):
        env = gym.make(env_name)
        if args.use_reward_wrapper and reward_config:
            env = RocketRewardWrapper(env, reward_config)
        set_seed(int(seed))
        return env
    
    # Evaluate
    print("Running evaluation...")
    metrics = compute_landing_metrics(
        agent,
        env_factory,
        test_seeds,
        num_episodes_per_seed=max(1, args.num_episodes // len(test_seeds))
    )
    
    # Print summary
    print()
    print_evaluation_summary(metrics, args.algorithm.upper())
    
    # Save results to CSV
    import pandas as pd
    import os
    
    # Extract optimizer from checkpoint path
    checkpoint_name = os.path.basename(args.checkpoint)
    optimizer = None
    for opt in ["adam", "rmsprop", "sgd"]:
        if opt in checkpoint_name.lower():
            optimizer = opt
            break
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Prepare results row
    results_row = {
        "algorithm": args.algorithm,
        "optimizer": optimizer if optimizer else "unknown",
        "checkpoint": checkpoint_name,
        "mean_return": metrics.get("mean_return", 0),
        "std_return": metrics.get("std_return", 0),
        "mean_episode_length": metrics.get("mean_episode_length", 0),
        "std_episode_length": metrics.get("std_episode_length", 0),
        "success_rate": metrics.get("success_rate", 0) * 100,  # Convert to percentage
        "crash_rate": metrics.get("crash_rate", 0) * 100,  # Convert to percentage
        "mean_fuel_usage": metrics.get("mean_fuel_usage", 0),
        "std_fuel_usage": metrics.get("std_fuel_usage", 0),
        "min_fuel_usage": metrics.get("min_fuel_usage", 0),
        "max_fuel_usage": metrics.get("max_fuel_usage", 0),
        "total_episodes": metrics.get("total_episodes", 0),
        "use_reward_wrapper": args.use_reward_wrapper
    }
    
    # Load existing results or create new DataFrame
    results_file = "results/test_results.csv"
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            # Check if CSV has the new format (with algorithm column)
            if "algorithm" in df.columns:
                # Check if this exact checkpoint already exists
                existing = df[(df["algorithm"] == args.algorithm) & 
                             (df["checkpoint"] == checkpoint_name) &
                             (df["use_reward_wrapper"] == args.use_reward_wrapper)]
                if len(existing) > 0:
                    # Update existing row
                    idx = existing.index[0]
                    for key, value in results_row.items():
                        df.at[idx, key] = value
                else:
                    # Append new row
                    df = pd.concat([df, pd.DataFrame([results_row])], ignore_index=True)
            else:
                # Old format detected - create new DataFrame with new format
                # (Old data will be preserved in backup if needed, but we start fresh with new format)
                df = pd.DataFrame([results_row])
        except Exception as e:
            # If there's any error reading the file, create new DataFrame
            print(f"Warning: Could not read existing results file: {e}")
            df = pd.DataFrame([results_row])
    else:
        # Create new DataFrame
        df = pd.DataFrame([results_row])
    
    # Save to CSV
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    print("="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

