"""
Hyperparameter tuning framework with grid search and CSV logging.
Supports systematic sweeps over key hyperparameters.
"""

import os
import itertools
import pandas as pd
from typing import Dict, List, Any, Callable, Optional
from utils.config import RewardConfig, NetworkConfig, OptimizerConfig, TrainingConfig
from training.trainer import train_agent, evaluate_on_seeds
from utils.plotting import plot_hyperparameter_sweep_results


def grid_search(
    param_grid: Dict[str, List[Any]],
    agent_factory: Callable,
    env_factory: Callable,
    train_config: TrainingConfig,
    val_seeds: List[int],
    results_dir: str = "hyperparameter_results"
) -> pd.DataFrame:
    """
    Perform grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        agent_factory: Function that creates agent given hyperparameters
        env_factory: Function that creates environment given seed
        train_config: Training configuration
        val_seeds: Validation seeds for evaluation
        results_dir: Directory to save results
        
    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    print(f"Starting grid search over {len(combinations)} combinations...")
    print(f"Parameters: {param_names}")
    
    for idx, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        print(f"\n[{idx + 1}/{len(combinations)}] Testing: {params}")
        
        # Create agent with these hyperparameters
        agent = agent_factory(**params)
        
        # Train agent
        train_stats = train_agent(
            agent,
            env_factory,
            train_config,
            agent_name=f"grid_search_{idx}",
            save_dir=os.path.join(results_dir, "checkpoints"),
            log_dir=os.path.join(results_dir, "logs"),
            plot_dir=os.path.join(results_dir, "plots")
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_on_seeds(
            agent, env_factory, val_seeds, num_episodes_per_seed=3
        )
        
        # Record results
        result = params.copy()
        result["val_return"] = val_metrics["mean_return"]
        result["val_std"] = val_metrics["std_return"]
        result["val_success_rate"] = val_metrics.get("success_rate", 0)
        result["val_fuel_usage"] = val_metrics.get("mean_fuel_usage", 0)
        result["best_train_return"] = max(train_stats["train_returns"]) if train_stats["train_returns"] else 0
        result["final_train_return"] = train_stats["train_returns"][-1] if train_stats["train_returns"] else 0
        
        results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(results_dir, "sweep_results.csv"), index=False)
        
        print(f"  Val Return: {val_metrics['mean_return']:.2f} ± {val_metrics['std_return']:.2f}")
    
    # Final results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "sweep_results_final.csv"), index=False)
    
    print(f"\nGrid search complete! Results saved to {results_dir}/")
    
    return results_df


def hyperparameter_sweep_dqn(
    env_factory: Callable,
    train_config: TrainingConfig,
    val_seeds: List[int],
    results_dir: str = "hyperparameter_results/dqn"
):
    """
    Perform hyperparameter sweep for DQN.
    
    Args:
        env_factory: Function that creates environment
        train_config: Training configuration
        val_seeds: Validation seeds
        results_dir: Directory to save results
    """
    from agents.dqn import DQNAgent
    from utils.device import get_device
    
    def agent_factory(**kwargs):
        # Default parameters
        defaults = {
            "state_dim": 8,  # LunarLander state dimension
            "action_dim": 4,  # LunarLander action dimension
            "network_config": NetworkConfig(
                hidden_sizes=kwargs.get("hidden_sizes", [128, 128]),
                dropout_rate=kwargs.get("dropout_rate", 0.0),
                use_dropout=kwargs.get("use_dropout", False)
            ),
            "optimizer_config": OptimizerConfig(
                optimizer=kwargs.get("optimizer", "adam"),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                weight_decay=kwargs.get("weight_decay", 0.0),
                use_scheduler=kwargs.get("use_scheduler", False)
            ),
            "gamma": kwargs.get("gamma", 0.99),
            "epsilon_start": kwargs.get("epsilon_start", 1.0),
            "epsilon_end": kwargs.get("epsilon_end", 0.01),
            "epsilon_decay": kwargs.get("epsilon_decay", 0.995),
            "device": get_device()
        }
        defaults.update(kwargs)
        return DQNAgent(**defaults)
    
    # Define parameter grid
    param_grid = {
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "gamma": [0.95, 0.99, 0.999],
        "optimizer": ["adam", "rmsprop"],
        "weight_decay": [0.0, 1e-5, 1e-4]
    }
    
    return grid_search(param_grid, agent_factory, env_factory, train_config, val_seeds, results_dir)


def hyperparameter_sweep_policy(
    env_factory: Callable,
    train_config: TrainingConfig,
    val_seeds: List[int],
    agent_type: str = "reinforce",  # "reinforce" or "a2c"
    results_dir: str = "hyperparameter_results"
):
    """
    Perform hyperparameter sweep for policy gradient methods.
    
    Args:
        env_factory: Function that creates environment
        train_config: Training configuration
        val_seeds: Validation seeds
        agent_type: Type of agent ("reinforce" or "a2c")
        results_dir: Directory to save results
    """
    from agents.reinforce import REINFORCEAgent
    from agents.a2c import A2CAgent
    from utils.device import get_device
    
    def agent_factory(**kwargs):
        defaults = {
            "state_dim": 8,
            "action_dim": 4,
            "network_config": NetworkConfig(
                hidden_sizes=kwargs.get("hidden_sizes", [128, 128])
            ),
            "optimizer_config": OptimizerConfig(
                optimizer=kwargs.get("optimizer", "adam"),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                weight_decay=kwargs.get("weight_decay", 0.0)
            ),
            "gamma": kwargs.get("gamma", 0.99),
            "entropy_coef": kwargs.get("entropy_coef", 0.01),
            "device": get_device()
        }
        defaults.update(kwargs)
        
        if agent_type == "reinforce":
            return REINFORCEAgent(**defaults)
        else:
            defaults["value_coef"] = kwargs.get("value_coef", 0.5)
            return A2CAgent(**defaults)
    
    # Define parameter grid
    param_grid = {
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.999],
        "entropy_coef": [0.001, 0.01, 0.1],
        "optimizer": ["adam", "sgd"]
    }
    
    if agent_type == "a2c":
        param_grid["value_coef"] = [0.25, 0.5, 1.0]
    
    results_dir = os.path.join(results_dir, agent_type)
    return grid_search(param_grid, agent_factory, env_factory, train_config, val_seeds, results_dir)


def hyperparameter_sweep_reward(
    env_factory_base: Callable,
    train_config: TrainingConfig,
    val_seeds: List[int],
    agent_factory: Callable,
    results_dir: str = "hyperparameter_results/reward"
):
    """
    Perform hyperparameter sweep over reward function parameters.
    
    Args:
        env_factory_base: Base environment factory (without reward wrapper)
        train_config: Training configuration
        val_seeds: Validation seeds
        agent_factory: Function that creates agent
        results_dir: Directory to save results
    """
    from environments.reward_wrapper import RocketRewardWrapper
    import gymnasium as gym
    
    def env_factory_with_reward(seed, reward_config):
        env = gym.make("LunarLander-v3")
        env = RocketRewardWrapper(env, reward_config)
        return env
    
    def agent_factory_with_reward(**kwargs):
        reward_config = RewardConfig(
            landing_bonus=kwargs.get("landing_bonus", 100.0),
            fuel_penalty=kwargs.get("fuel_penalty", 0.1),
            crash_penalty=kwargs.get("crash_penalty", -100.0),
            smoothness_penalty=kwargs.get("smoothness_penalty", 0.05)
        )
        return agent_factory(reward_config=reward_config)
    
    # Define parameter grid
    param_grid = {
        "landing_bonus": [50.0, 100.0, 200.0],
        "fuel_penalty": [0.05, 0.1, 0.2],
        "crash_penalty": [-50.0, -100.0, -200.0],
        "smoothness_penalty": [0.01, 0.05, 0.1]
    }
    
    # This would need to be adapted to work with the grid_search function
    # For now, return a simplified version
    print("Reward hyperparameter sweep - implement custom logic if needed")
    return None

