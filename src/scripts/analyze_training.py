# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create an analysis script that reads training CSV logs, computes moving averages, and generates learning curves"

"""
Analysis script for training logs.
Reads CSV logs, computes moving averages, and generates learning curves.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from utils.plotting import moving_average, plot_learning_curve


def analyze_training_log(csv_path: str, window: int = 100, save_plot: str = None):
    """
    Analyze training log CSV and generate learning curves.
    
    Args:
        csv_path: Path to CSV training log
        window: Window size for moving average
        save_plot: Path to save plot (optional)
    """
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("TRAINING LOG ANALYSIS")
    print("="*80)
    print(f"Log file: {csv_path}")
    print(f"Total episodes: {len(df)}")
    print()
    
    # Basic statistics
    print("Return Statistics:")
    print(f"  Mean: {df['return'].mean():.2f}")
    print(f"  Std: {df['return'].std():.2f}")
    print(f"  Min: {df['return'].min():.2f}")
    print(f"  Max: {df['return'].max():.2f}")
    print(f"  First 10 episodes avg: {df['return'].head(10).mean():.2f}")
    print(f"  Last 10 episodes avg: {df['return'].tail(10).mean():.2f}")
    print()
    
    print("Episode Length Statistics:")
    print(f"  Mean: {df['length'].mean():.1f}")
    print(f"  Std: {df['length'].std():.1f}")
    print(f"  Min: {df['length'].min():.0f}")
    print(f"  Max: {df['length'].max():.0f}")
    print()
    
    # Epsilon statistics (if available)
    if 'epsilon' in df.columns and df['epsilon'].notna().any():
        epsilon_data = df['epsilon'].dropna()
        if len(epsilon_data) > 0:
            print("Epsilon Statistics:")
            print(f"  Initial: {epsilon_data.iloc[0]:.4f}")
            print(f"  Final: {epsilon_data.iloc[-1]:.4f}")
            print()
    
    # Learning rate statistics (if available)
    if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
        lr_data = df['learning_rate'].dropna()
        if len(lr_data) > 0:
            print("Learning Rate Statistics:")
            print(f"  Initial: {lr_data.iloc[0]:.6f}")
            print(f"  Final: {lr_data.iloc[-1]:.6f}")
            if len(lr_data.unique()) > 1:
                print(f"  Changed: Yes (scheduler active)")
            print()
    
    # Validation statistics (if available)
    if 'val_return' in df.columns and df['val_return'].notna().any():
        val_data = df['val_return'].dropna()
        if len(val_data) > 0:
            print("Validation Return Statistics:")
            print(f"  Mean: {val_data.mean():.2f}")
            print(f"  Std: {val_data.std():.2f}")
            print(f"  Best: {val_data.max():.2f}")
            print(f"  Best episode: {df.loc[val_data.idxmax(), 'episode']:.0f}")
            print()
    
    # Compute moving average
    returns = df['return'].values
    ma_returns = moving_average(returns, window)
    ma_episodes = np.arange(window - 1, len(returns))
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Raw returns
    plt.plot(df['episode'], df['return'], alpha=0.3, color='blue', label='Raw Returns')
    
    # Moving average
    if len(ma_returns) > 0:
        plt.plot(ma_episodes + 1, ma_returns, color='red', linewidth=2, 
                label=f'Moving Average (window={window})')
    
    # Validation returns if available
    if 'val_return' in df.columns and df['val_return'].notna().any():
        val_episodes = df[df['val_return'].notna()]['episode']
        val_returns = df[df['val_return'].notna()]['val_return']
        plt.scatter(val_episodes, val_returns, color='green', alpha=0.5, 
                   s=20, label='Validation Returns', zorder=5)
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Learning Curve: {os.path.basename(csv_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        os.makedirs(os.path.dirname(save_plot) if os.path.dirname(save_plot) else '.', exist_ok=True)
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_plot}")
    else:
        plt.show()
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log", type=str, required=True, 
                       help="Path to training log CSV file")
    parser.add_argument("--window", type=int, default=100,
                       help="Window size for moving average")
    parser.add_argument("--save_plot", type=str, default=None,
                       help="Path to save plot (optional)")
    
    args = parser.parse_args()
    
    analyze_training_log(args.log, args.window, args.save_plot)


if __name__ == "__main__":
    main()

