"""
Generate validation and test metrics comparison plots for all four models.
"""

import os
import sys
import torch

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from utils.device import get_device
from utils.config import NetworkConfig, OptimizerConfig, TrainingConfig
from training.trainer import create_env_factory
from environments.reward_wrapper import RocketRewardWrapper
from evaluation.compare_agents import compare_all_agents


def load_agent(algorithm: str, optimizer: str, checkpoint_path: str):
    """
    Load agent from checkpoint.
    
    Args:
        algorithm: Algorithm name ("dqn", "a2c")
        optimizer: Optimizer name ("adam", "rmsprop", "sgd")
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded agent
    """
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
        return agent
    
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
        return agent
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Generate validation and test metrics comparison plots."""
    
    # Define all models to compare
    models = [
        ("DQN (Adam)", "dqn", "adam", "models/dqn/dqn_adam_best.pt"),
        ("DQN (RMSprop)", "dqn", "rmsprop", "models/dqn/dqn_rmsprop_best.pt"),
        ("A2C (Adam)", "a2c", "adam", "models/a2c/a2c_adam_best.pt"),
        ("A2C (SGD)", "a2c", "sgd", "models/a2c/a2c_sgd_best.pt"),
    ]
    
    # Load all agents
    agents = {}
    for name, algorithm, optimizer, checkpoint_path in models:
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"Loading {name} from {checkpoint_path}...")
        try:
            agent = load_agent(algorithm, optimizer, checkpoint_path)
            agents[name] = agent
            print(f"  ✓ Successfully loaded {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            continue
    
    if not agents:
        print("Error: No agents loaded. Please check checkpoint paths.")
        return
    
    print(f"\nLoaded {len(agents)} agents: {list(agents.keys())}")
    
    # Create environment factory with reward wrapper
    def env_factory(seed=None):
        import gymnasium as gym
        env = gym.make("LunarLander-v3")
        if seed is not None:
            env.reset(seed=seed)
        # Use reward wrapper (matches training setup)
        from utils.config import RewardConfig
        reward_config = RewardConfig()
        env = RocketRewardWrapper(env, reward_config)
        return env
    
    # Define validation and test seeds (matching training config)
    val_seeds = list(range(100, 110))  # 10 validation seeds
    test_seeds = list(range(200, 210))  # 10 test seeds
    
    print(f"\nValidation seeds: {val_seeds}")
    print(f"Test seeds: {test_seeds}")
    
    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING METRICS COMPARISON PLOTS")
    print("="*80)
    
    results = compare_all_agents(
        agents=agents,
        env_factory=env_factory,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        plot_dir="data/plots",
        results_dir="results"
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Validation metrics plot: data/plots/validation_metrics_comparison.png")
    print(f"Test metrics plot: data/plots/test_metrics_comparison.png")
    print(f"Results saved to: results/validation_results.csv and results/test_results.csv")


if __name__ == "__main__":
    main()

