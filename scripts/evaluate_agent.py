"""
Evaluation entry point for trained agents.
Loads best checkpoint and evaluates on held-out test seeds.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from training.trainer import set_seed, create_env_factory
from environments.reward_wrapper import RocketRewardWrapper
from utils.config import RewardConfig, TrainingConfig
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
        agent = A2CAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
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
    
    print("="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

