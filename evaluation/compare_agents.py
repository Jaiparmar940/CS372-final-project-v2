# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create a script to compare multiple trained RL agents and generate comparison plots showing performance metrics"

"""
Compare multiple RL agents and generate comparison plots.
"""

import os
import numpy as np
from typing import Dict, Any, List
import pandas as pd

from evaluation.evaluator import compare_agents, print_evaluation_summary
from utils.plotting import plot_comparison, plot_metrics_comparison


def compare_all_agents(
    agents: Dict[str, Any],
    env_factory,
    val_seeds: List[int],
    test_seeds: List[int],
    plot_dir: str = "plots",
    results_dir: str = "results"
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare all agents on validation and test sets.
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        env_factory: Function that creates environment given seed
        val_seeds: Validation seeds
        test_seeds: Test seeds
        plot_dir: Directory to save plots
        results_dir: Directory to save results
        
    Returns:
        Dictionary with validation and test results
    """
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_results = compare_agents(agents, env_factory, val_seeds, num_episodes_per_seed=5)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = compare_agents(agents, env_factory, test_seeds, num_episodes_per_seed=5)
    
    # Print summaries
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    for agent_name, metrics in val_results.items():
        print_evaluation_summary(metrics, agent_name)
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    for agent_name, metrics in test_results.items():
        print_evaluation_summary(metrics, agent_name)
    
    # Generate comparison plots
    plot_metrics_comparison(
        val_results,
        title="Validation Metrics Comparison",
        save_path=os.path.join(plot_dir, "validation_metrics_comparison.png")
    )
    
    plot_metrics_comparison(
        test_results,
        title="Test Metrics Comparison",
        save_path=os.path.join(plot_dir, "test_metrics_comparison.png")
    )
    
    # Save results to CSV
    save_comparison_results(val_results, test_results, results_dir)
    
    return {
        "validation": val_results,
        "test": test_results
    }


def save_comparison_results(
    val_results: Dict[str, Dict[str, float]],
    test_results: Dict[str, Dict[str, float]],
    results_dir: str
):
    """
    Save comparison results to CSV files.
    
    Args:
        val_results: Validation results
        test_results: Test results
        results_dir: Directory to save results
    """
    # Validation results
    val_df = pd.DataFrame(val_results).T
    val_df.to_csv(os.path.join(results_dir, "validation_results.csv"))
    
    # Test results
    test_df = pd.DataFrame(test_results).T
    test_df.to_csv(os.path.join(results_dir, "test_results.csv"))
    
    print(f"\nResults saved to {results_dir}/")


def plot_learning_curves_comparison(
    agents: Dict[str, Any],
    plot_dir: str = "plots"
):
    """
    Plot learning curves for all agents.
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        plot_dir: Directory to save plots
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # Collect episode returns from all agents
    learning_curves = {}
    
    for agent_name, agent in agents.items():
        if hasattr(agent, 'episode_returns'):
            learning_curves[agent_name] = agent.episode_returns
    
    if learning_curves:
        plot_comparison(
            learning_curves,
            title="Learning Curves Comparison",
            save_path=os.path.join(plot_dir, "learning_curves_comparison.png")
        )

