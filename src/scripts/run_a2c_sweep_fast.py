"""
Fast A2C hyperparameter sweep script.

Run examples:
  python run_a2c_sweep_fast.py --fast --reduced-grid --episodes 300
  python run_a2c_sweep_fast.py --fast --episodes 300
  python run_a2c_sweep_fast.py --episodes 500
"""

import os
import sys
import gymnasium as gym

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from hyperparameter_tuning.sweep import hyperparameter_sweep_policy
from training.trainer import set_seed
from utils.config import TrainingConfig, RewardConfig
from environments.reward_wrapper import RocketRewardWrapper


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run fast A2C hyperparameter sweep")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per combination (default: 300)")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (reduces validation overhead)")
    parser.add_argument("--reduced-grid", action="store_true", help="Use reduced hyperparameter grid")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Training configuration optimized for sweep
    config = TrainingConfig(
        num_episodes=args.episodes,
        train_seeds=list(range(42, 52)),
        val_seeds=list(range(100, 110)),
        test_seeds=list(range(200, 210)),
        val_frequency=30,  # Validate less frequently for speed
        val_episodes_per_seed=1 if args.fast else 3,
        early_stopping=True,
        patience=30,
        min_delta=0.01
    )

    reward_config = RewardConfig()

    def env_factory(seed):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env

    print("=" * 80)
    print("A2C Hyperparameter Sweep")
    print("=" * 80)
    print(f"Episodes per combination: {config.num_episodes}")
    print(f"Fast mode: {args.fast}")
    print(f"Reduced grid: {args.reduced_grid}")

    if args.reduced_grid:
        print("\nUsing REDUCED grid (smaller search for speed)")
    else:
        print("\nUsing FULL grid (more coverage)")

    results = hyperparameter_sweep_policy(
        env_factory=env_factory,
        train_config=config,
        val_seeds=list(range(100, 110)),
        agent_type="a2c",
        results_dir="hyperparameter_results",
        fast_mode=args.fast,
        reduced_grid=args.reduced_grid
    )

    print("\n" + "=" * 80)
    print("Sweep Complete!")
    print("=" * 80)
    print("\nBest hyperparameters:")
    best_idx = results["val_return"].idxmax()
    best_params = results.loc[best_idx]
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Gamma: {best_params['gamma']}")
    print(f"  Entropy coef: {best_params['entropy_coef']}")
    if "value_coef" in best_params:
        print(f"  Value coef: {best_params['value_coef']}")
    print(f"  Optimizer: {best_params['optimizer']}")
    print(f"  Validation return: {best_params['val_return']:.2f} ± {best_params['val_std']:.2f}")
    print(f"\nResults saved to: hyperparameter_results/a2c/")


if __name__ == "__main__":
    main()

