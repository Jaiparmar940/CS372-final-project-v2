"""
Fast DQN hyperparameter sweep script.

This script demonstrates how to run an optimized hyperparameter sweep.
"""

import os
import sys
import gymnasium as gym

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from hyperparameter_tuning.sweep import hyperparameter_sweep_dqn
from training.trainer import create_env_factory, set_seed
from utils.config import TrainingConfig, RewardConfig
from environments.reward_wrapper import RocketRewardWrapper


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run fast DQN hyperparameter sweep")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes per combination (default: 300)")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (reduces validation overhead)")
    parser.add_argument("--reduced-grid", action="store_true", help="Use reduced hyperparameter grid (24 vs 72 combinations)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Training configuration optimized for hyperparameter search
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210)),
        val_frequency=30,  # Validate every 30 episodes (less frequent)
        val_episodes_per_seed=1 if args.fast else 3,
        early_stopping=True,
        patience=30,  # More aggressive early stopping
        min_delta=0.01
    )
    
    # Reward configuration
    reward_config = RewardConfig()
    
    # Create environment factory
    def env_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    # Run sweep
    print("="*80)
    print("DQN Hyperparameter Sweep")
    print("="*80)
    print(f"Episodes per combination: {config.num_episodes}")
    print(f"Fast mode: {args.fast}")
    print(f"Reduced grid: {args.reduced_grid}")
    
    if args.reduced_grid:
        print("\nUsing REDUCED grid: 16 combinations (vs 72)")
    else:
        print("\nUsing FULL grid: 72 combinations")
    
    results = hyperparameter_sweep_dqn(
        env_factory=env_factory,
        train_config=config,
        val_seeds=list(range(100, 110)),
        results_dir="hyperparameter_results/dqn",
        fast_mode=args.fast,
        reduced_grid=args.reduced_grid
    )
    
    print("\n" + "="*80)
    print("Sweep Complete!")
    print("="*80)
    print(f"\nBest hyperparameters:")
    best_idx = results["val_return"].idxmax()
    best_params = results.loc[best_idx]
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Gamma: {best_params['gamma']}")
    print(f"  Optimizer: {best_params['optimizer']}")
    print(f"  Weight decay: {best_params['weight_decay']}")
    print(f"  Validation return: {best_params['val_return']:.2f} ± {best_params['val_std']:.2f}")
    print(f"\nResults saved to: hyperparameter_results/dqn/")


if __name__ == "__main__":
    main()

