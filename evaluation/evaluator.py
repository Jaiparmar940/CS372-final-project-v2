"""
Evaluation utilities for RL agents.
Computes success rate, fuel usage, and other metrics on validation/test sets.
"""

import numpy as np
from typing import Dict, List, Any, Callable
from training.trainer import evaluate_on_seeds


def evaluate_agent(
    agent: Any,
    env_factory: Callable,
    seeds: List[int],
    num_episodes_per_seed: int = 5
) -> Dict[str, float]:
    """
    Comprehensive evaluation of agent on set of seeds.
    
    Args:
        agent: RL agent to evaluate
        env_factory: Function that creates environment given seed
        seeds: List of seeds to evaluate on
        num_episodes_per_seed: Number of episodes per seed
        
    Returns:
        Dictionary of evaluation metrics
    """
    return evaluate_on_seeds(agent, env_factory, seeds, num_episodes_per_seed)


def compute_landing_metrics(
    agent: Any,
    env_factory: Callable,
    seeds: List[int],
    num_episodes_per_seed: int = 5
) -> Dict[str, float]:
    """
    Compute metrics specifically related to rocket landing task.
    
    Args:
        agent: RL agent to evaluate
        env_factory: Function that creates environment given seed
        seeds: List of seeds to evaluate on
        num_episodes_per_seed: Number of episodes per seed
        
    Returns:
        Dictionary of landing-specific metrics
    """
    all_returns = []
    all_lengths = []
    success_count = 0
    crash_count = 0
    fuel_usage = []
    total_episodes = 0
    
    for seed in seeds:
        env = env_factory(seed)
        
        for _ in range(num_episodes_per_seed):
            metrics = agent.evaluate(env, num_episodes=1, seed=seed)
            all_returns.append(metrics["mean_return"])
            all_lengths.append(metrics["mean_length"])
            
            if metrics.get("success_rate", 0) > 0:
                success_count += 1
            if metrics.get("crash_rate", 0) > 0:
                crash_count += 1
            
            if "mean_fuel_usage" in metrics:
                fuel_usage.append(metrics["mean_fuel_usage"])
            
            total_episodes += 1
    
    results = {
        "mean_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "mean_episode_length": np.mean(all_lengths),
        "std_episode_length": np.std(all_lengths),
        "success_rate": success_count / total_episodes if total_episodes > 0 else 0.0,
        "crash_rate": crash_count / total_episodes if total_episodes > 0 else 0.0,
        "total_episodes": total_episodes
    }
    
    if fuel_usage:
        results["mean_fuel_usage"] = np.mean(fuel_usage)
        results["std_fuel_usage"] = np.std(fuel_usage)
        results["min_fuel_usage"] = np.min(fuel_usage)
        results["max_fuel_usage"] = np.max(fuel_usage)
    
    return results


def compare_agents(
    agents: Dict[str, Any],
    env_factory: Callable,
    seeds: List[int],
    num_episodes_per_seed: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple agents on the same set of seeds.
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        env_factory: Function that creates environment given seed
        seeds: List of seeds to evaluate on
        num_episodes_per_seed: Number of episodes per seed
        
    Returns:
        Dictionary mapping agent names to their evaluation metrics
    """
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name}...")
        metrics = compute_landing_metrics(agent, env_factory, seeds, num_episodes_per_seed)
        results[agent_name] = metrics
    
    return results


def print_evaluation_summary(metrics: Dict[str, float], agent_name: str = "Agent"):
    """
    Print formatted evaluation summary.
    
    Args:
        metrics: Dictionary of evaluation metrics
        agent_name: Name of agent
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {agent_name}")
    print(f"{'='*60}")
    print(f"Mean Return: {metrics.get('mean_return', 0):.2f} ± {metrics.get('std_return', 0):.2f}")
    print(f"Mean Episode Length: {metrics.get('mean_episode_length', 0):.2f} ± {metrics.get('std_episode_length', 0):.2f}")
    print(f"Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")
    print(f"Crash Rate: {metrics.get('crash_rate', 0)*100:.1f}%")
    
    if "mean_fuel_usage" in metrics:
        print(f"Mean Fuel Usage: {metrics['mean_fuel_usage']:.2f} ± {metrics.get('std_fuel_usage', 0):.2f}")
        print(f"Fuel Usage Range: [{metrics.get('min_fuel_usage', 0):.2f}, {metrics.get('max_fuel_usage', 0):.2f}]")
    
    print(f"{'='*60}\n")

