"""
Plotting utilities for RL experiments.
Generates learning curves, comparison plots, and evaluation metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd


def moving_average(data: List[float], window: int = 100) -> np.ndarray:
    """
    Compute moving average of data.
    
    Args:
        data: List of values
        window: Window size for moving average
        
    Returns:
        Array of moving averages
    """
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_learning_curve(
    episode_returns: List[float],
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
    window: int = 100,
    show_moving_avg: bool = True,
    xlabel: str = "Episode",
    ylabel: str = "Episode Return"
):
    """
    Plot learning curve with optional moving average.
    
    Args:
        episode_returns: List of episode returns
        title: Plot title
        save_path: Path to save plot (optional)
        window: Window size for moving average
        show_moving_avg: Whether to show moving average
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(episode_returns))
    
    plt.plot(episodes, episode_returns, alpha=0.3, color='blue', label='Raw Returns')
    
    if show_moving_avg and len(episode_returns) >= window:
        ma = moving_average(episode_returns, window)
        ma_episodes = np.arange(window - 1, len(episode_returns))
        plt.plot(ma_episodes, ma, color='red', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    window: int = 100,
    xlabel: str = "Episode",
    ylabel: str = "Episode Return"
):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        results: Dictionary mapping algorithm names to episode returns
        title: Plot title
        save_path: Path to save plot (optional)
        window: Window size for moving average
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (name, returns) in enumerate(results.items()):
        episodes = np.arange(len(returns))
        
        # Plot raw returns with low alpha
        plt.plot(episodes, returns, alpha=0.2, color=colors[idx % len(colors)])
        
        # Plot moving average
        if len(returns) >= window:
            ma = moving_average(returns, window)
            ma_episodes = np.arange(window - 1, len(returns))
            plt.plot(ma_episodes, ma, color=colors[idx % len(colors)], 
                    linewidth=2, label=name)
        else:
            plt.plot(episodes, returns, color=colors[idx % len(colors)], 
                    linewidth=2, label=name)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Metrics Comparison",
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing metrics across algorithms.
    
    Args:
        metrics: Dictionary mapping algorithm names to metric dictionaries
        title: Plot title
        save_path: Path to save plot (optional)
    """
    if not metrics:
        return
    
    # Get all unique metric names
    all_metrics = set()
    for algo_metrics in metrics.values():
        all_metrics.update(algo_metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    if not all_metrics:
        return
    
    n_metrics = len(all_metrics)
    n_algos = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    algo_names = list(metrics.keys())
    x = np.arange(len(algo_names))
    width = 0.6
    
    for idx, metric_name in enumerate(all_metrics):
        values = [metrics[algo].get(metric_name, 0) for algo in algo_names]
        axes[idx].bar(x, values, width)
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(metric_name)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(algo_names, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.close()


def plot_hyperparameter_sweep_results(
    results_df: pd.DataFrame,
    param_name: str,
    metric_name: str = "val_return",
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot hyperparameter sweep results.
    
    Args:
        results_df: DataFrame with hyperparameter sweep results
        param_name: Name of parameter to plot on x-axis
        metric_name: Name of metric to plot on y-axis
        title: Plot title (optional)
        save_path: Path to save plot (optional)
    """
    if param_name not in results_df.columns or metric_name not in results_df.columns:
        print(f"Warning: {param_name} or {metric_name} not in results DataFrame")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by parameter value and compute mean/std
    grouped = results_df.groupby(param_name)[metric_name].agg(['mean', 'std']).reset_index()
    
    plt.errorbar(grouped[param_name], grouped['mean'], yerr=grouped['std'], 
                marker='o', capsize=5, capthick=2)
    plt.xlabel(param_name)
    plt.ylabel(metric_name)
    plt.title(title or f"{metric_name} vs {param_name}")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hyperparameter plot to {save_path}")
    
    plt.close()

