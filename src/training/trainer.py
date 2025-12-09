# This file was created by Cursor and fine tuned by Jaivir Parmar and Ryan Christ.
# To recreate this file, prompt Cursor with: "Implement base training utilities with seed management for train/val/test splits, early stopping based on validation performance, and model checkpointing"

"""
Base training utilities with seed management, early stopping, and checkpointing.
Provides common functionality for training all RL agents.
"""

import os
import numpy as np
import random
import torch
import csv
from typing import List, Optional, Dict, Callable, Any
from utils.config import TrainingConfig
from utils.plotting import plot_learning_curve


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping based on validation performance.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.01,
        mode: str = "max"  # "max" for maximizing metric, "min" for minimizing
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min" depending on whether metric should be maximized or minimized
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def print_config_summary(
    algorithm_name: str,
    environment_name: str,
    config: TrainingConfig,
    agent: Any,
    agent_name: str,
    reward_config: Any = None
):
    """
    Print a clear CONFIG SUMMARY block at the start of training.
    
    Args:
        algorithm_name: Name of the algorithm
        environment_name: Name of the environment
        config: Training configuration
        agent: Agent instance to extract hyperparameters from
        agent_name: Name identifier for the agent
    """
    print("\n" + "="*80)
    print("CONFIG SUMMARY")
    print("="*80)
    print(f"Algorithm: {algorithm_name}")
    print(f"Environment: {environment_name}")
    print(f"Agent Name: {agent_name}")
    print()
    print("Training Configuration:")
    print(f"  num_episodes: {config.num_episodes}")
    print(f"  max_steps_per_episode: {config.max_steps_per_episode}")
    print(f"  gamma (discount factor): {config.gamma}")
    print(f"  train_seeds: {config.train_seeds[:5]}... (first 5 of {len(config.train_seeds)})")
    print(f"  val_seeds: {config.val_seeds[:5]}... (first 5 of {len(config.val_seeds)})")
    print(f"  test_seeds: {config.test_seeds[:5]}... (first 5 of {len(config.test_seeds)})")
    print()
    print("Agent Hyperparameters:")
    
    # Extract hyperparameters based on agent type
    if hasattr(agent, 'learning_rate'):
        print(f"  learning_rate: {agent.learning_rate}")
    if hasattr(agent, 'gamma'):
        print(f"  gamma: {agent.gamma}")
    if hasattr(agent, 'epsilon') or hasattr(agent, 'epsilon_start'):
        if hasattr(agent, 'epsilon_start'):
            print(f"  epsilon_start: {agent.epsilon_start}")
            print(f"  epsilon_end: {agent.epsilon_end}")
            print(f"  epsilon_decay: {agent.epsilon_decay}")
        elif hasattr(agent, 'epsilon'):
            print(f"  epsilon: {agent.epsilon}")
    if hasattr(agent, 'batch_size'):
        print(f"  batch_size: {agent.batch_size}")
    if hasattr(agent, 'replay_buffer_size'):
        print(f"  replay_buffer_size: {agent.replay_buffer_size}")
    if hasattr(agent, 'target_update_frequency'):
        print(f"  target_update_frequency: {agent.target_update_frequency}")
    if hasattr(agent, 'entropy_coef'):
        print(f"  entropy_coef: {agent.entropy_coef}")
    if hasattr(agent, 'optimizer_config'):
        opt_config = agent.optimizer_config
        print(f"  optimizer: {opt_config.optimizer}")
        print(f"  optimizer_lr: {opt_config.learning_rate}")
        print(f"  weight_decay: {opt_config.weight_decay}")
        if opt_config.use_scheduler:
            print(f"  scheduler_type: {opt_config.scheduler_type}")
    if hasattr(agent, 'optimizer'):
        # Try to get learning rate from optimizer
        try:
            lr = agent.optimizer.param_groups[0]['lr']
            print(f"  current_learning_rate: {lr}")
        except:
            pass
    
    # Reward configuration (if using wrapper)
    if reward_config is not None:
        print()
        print("Reward Configuration:")
        if hasattr(reward_config, 'landing_bonus'):
            print(f"  landing_bonus: {reward_config.landing_bonus}")
        if hasattr(reward_config, 'fuel_penalty'):
            print(f"  fuel_penalty: {reward_config.fuel_penalty}")
        if hasattr(reward_config, 'crash_penalty'):
            print(f"  crash_penalty: {reward_config.crash_penalty}")
        if hasattr(reward_config, 'smoothness_penalty'):
            print(f"  smoothness_penalty: {reward_config.smoothness_penalty}")
        if hasattr(reward_config, 'base_reward_scale'):
            print(f"  base_reward_scale: {reward_config.base_reward_scale}")
    
    print()
    print("="*80)
    print()


def train_agent(
    agent: Any,
    env_factory: Callable,
    config: TrainingConfig,
    agent_name: str,
    algorithm_name: str = None,
    environment_name: str = None,
    reward_config: Any = None,
    save_dir: str = "checkpoints",
    log_dir: str = "data/logs",
    plot_dir: str = "data/plots"
) -> Dict[str, Any]:
    """
    Generic training function for RL agents.
    
    Args:
        agent: RL agent to train
        env_factory: Function that creates environment given seed
        config: Training configuration
        agent_name: Name of agent (for saving)
        save_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with training results and statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Print config summary
    if algorithm_name and environment_name:
        print_config_summary(algorithm_name, environment_name, config, agent, agent_name, reward_config)
    
    # CSV logging setup
    csv_path = os.path.join(log_dir, f"{agent_name}_training_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'return', 'length', 'epsilon', 'learning_rate', 'val_return'])
    
    # Early stopping
    early_stopping = None
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
    
    # Training statistics
    train_returns = []
    val_returns = []
    best_val_return = float('-inf')
    best_episode = 0
    recent_episodes = []  # Store last 50 episodes for console output
    
    # For robust model selection: track moving average of validation returns
    val_return_window = []  # Store recent validation returns for moving average
    window_size = 5  # Use average of last 5 validation checks for model selection
    
    for episode in range(config.num_episodes):
        # Train on random training seed
        train_seed = int(np.random.choice(config.train_seeds))
        env = env_factory(train_seed)
        
        # Train one episode
        episode_return, episode_length, episode_info = agent.train_episode(
            env, max_steps=config.max_steps_per_episode
        )
        train_returns.append(episode_return)
        
        # Get current epsilon and learning rate
        current_epsilon = None
        if hasattr(agent, 'epsilon'):
            current_epsilon = agent.epsilon
        elif hasattr(agent, 'epsilon_history') and agent.epsilon_history:
            current_epsilon = agent.epsilon_history[-1]
        
        current_lr = None
        if hasattr(agent, 'optimizer'):
            try:
                current_lr = agent.optimizer.param_groups[0]['lr']
            except:
                pass
        
        # Validation evaluation
        val_return = None
        val_frequency = getattr(config, 'val_frequency', config.log_frequency)
        val_episodes_per_seed = getattr(config, 'val_episodes_per_seed', 5)
        if (episode + 1) % val_frequency == 0:
            # Use more episodes per seed to reduce noise in validation metrics
            val_metrics = evaluate_on_seeds(
                agent, env_factory, config.val_seeds, num_episodes_per_seed=val_episodes_per_seed
            )
            val_return = val_metrics["mean_return"]
            val_returns.append(val_return)
            
            # Track validation returns for moving average
            val_return_window.append(val_return)
            if len(val_return_window) > window_size:
                val_return_window.pop(0)
            
            # Update learning rate scheduler if available
            if hasattr(agent, 'update_scheduler'):
                agent.update_scheduler(val_return)
                # Update current_lr after scheduler step
                if hasattr(agent, 'optimizer'):
                    try:
                        current_lr = agent.optimizer.param_groups[0]['lr']
                    except:
                        pass
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_return):
                    print(f"Early stopping at episode {episode + 1}")
                    break
            
            # Save best model using moving average for more robust selection
            # Only update if we have enough validation checks for a meaningful average
            if len(val_return_window) >= window_size:
                avg_val_return = np.mean(val_return_window)
                # Use moving average for model selection to reduce noise
                if avg_val_return > best_val_return:
                    best_val_return = avg_val_return
                    best_episode = episode + 1
                    checkpoint_path = os.path.join(save_dir, f"{agent_name}_best.pt")
                    if hasattr(agent, 'save'):
                        agent.save(checkpoint_path)
            else:
                # For early episodes, use single validation return
                if val_return > best_val_return:
                    best_val_return = val_return
                    best_episode = episode + 1
                    checkpoint_path = os.path.join(save_dir, f"{agent_name}_best.pt")
                    if hasattr(agent, 'save'):
                        agent.save(checkpoint_path)
        
        # Log to CSV (every episode)
        csv_writer.writerow([
            episode + 1,
            episode_return,
            episode_length,
            current_epsilon if current_epsilon is not None else '',
            current_lr if current_lr is not None else '',
            val_return if val_return is not None else ''
        ])
        csv_file.flush()
        
        # Store recent episodes for console output
        recent_episodes.append({
            'episode': episode + 1,
            'return': episode_return,
            'length': episode_length,
            'epsilon': current_epsilon,
            'lr': current_lr,
            'val_return': val_return
        })
        if len(recent_episodes) > 50:
            recent_episodes.pop(0)
        
        # Console logging (every log_frequency episodes or last 50)
        if (episode + 1) % config.log_frequency == 0 or episode >= config.num_episodes - 50:
            epsilon_str = f"{current_epsilon:.4f}" if current_epsilon is not None else "N/A"
            lr_str = f"{current_lr:.6f}" if current_lr is not None else "N/A"
            
            if val_return is not None:
                print(f"Episode {episode + 1:5d}/{config.num_episodes} | "
                      f"Return: {episode_return:8.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Epsilon: {epsilon_str:>8} | "
                      f"LR: {lr_str:>10} | "
                      f"Val: {val_return:8.2f} | "
                      f"Best: {best_val_return:8.2f} (ep {best_episode})")
            else:
                print(f"Episode {episode + 1:5d}/{config.num_episodes} | "
                      f"Return: {episode_return:8.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Epsilon: {epsilon_str:>8} | "
                      f"LR: {lr_str:>10}")
        
        # Periodic checkpointing
        if (episode + 1) % config.save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"{agent_name}_ep{episode + 1}.pt")
            if hasattr(agent, 'save'):
                agent.save(checkpoint_path)
    
    csv_file.close()
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f"{agent_name}_final.pt")
    if hasattr(agent, 'save'):
        agent.save(final_checkpoint_path)
    
    # Plot learning curve
    plot_path = os.path.join(plot_dir, f"{agent_name}_learning_curve.png")
    plot_learning_curve(
        train_returns,
        title=f"{agent_name} Learning Curve",
        save_path=plot_path,
        window=100
    )
    
    # Print last 50 episodes summary
    print("\n" + "="*80)
    print("LAST 50 EPISODES SUMMARY")
    print("="*80)
    print(f"{'Episode':<10} {'Return':<12} {'Length':<10} {'Epsilon':<12} {'LR':<12} {'Val Return':<12}")
    print("-"*80)
    for ep in recent_episodes[-50:]:
        epsilon_str = f"{ep['epsilon']:.4f}" if ep['epsilon'] is not None else "N/A"
        lr_str = f"{ep['lr']:.6f}" if ep['lr'] is not None else "N/A"
        val_str = f"{ep['val_return']:.2f}" if ep['val_return'] is not None else "N/A"
        print(f"{ep['episode']:<10} {ep['return']:<12.2f} {ep['length']:<10} "
              f"{epsilon_str:<12} {lr_str:<12} {val_str:<12}")
    print("="*80)
    print(f"\nTraining log saved to: {csv_path}")
    print()
    
    # Save training statistics
    stats = {
        "train_returns": train_returns,
        "val_returns": val_returns,
        "best_val_return": best_val_return,
        "best_episode": best_episode,
        "total_episodes": len(train_returns),
        "csv_path": csv_path
    }
    
    return stats


def evaluate_on_seeds(
    agent: Any,
    env_factory: Callable,
    seeds: List[int],
    num_episodes_per_seed: int = 1
) -> Dict[str, float]:
    """
    Evaluate agent on multiple seeds.
    
    Args:
        agent: RL agent to evaluate
        env_factory: Function that creates environment given seed
        seeds: List of seeds to evaluate on
        num_episodes_per_seed: Number of episodes per seed
        
    Returns:
        Dictionary of aggregated evaluation metrics
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
        "mean_length": np.mean(all_lengths),
        "success_rate": success_count / total_episodes if total_episodes > 0 else 0.0,
        "crash_rate": crash_count / total_episodes if total_episodes > 0 else 0.0
    }
    
    if fuel_usage:
        results["mean_fuel_usage"] = np.mean(fuel_usage)
        results["std_fuel_usage"] = np.std(fuel_usage)
    
    return results


def create_env_factory(
    env_name: str,
    reward_config=None,
    use_wrapper: bool = True
) -> Callable:
    """
    Create environment factory function.
    
    Args:
        env_name: Name of environment (e.g., "LunarLander-v3")
        reward_config: Reward configuration for wrapper
        use_wrapper: Whether to use reward wrapper (for LunarLander)
        
    Returns:
        Function that creates environment given seed
    """
    # Assume Gymnasium environment
    import gymnasium as gym
    from environments.reward_wrapper import RocketRewardWrapper
    
    def factory(seed: int):
        env = gym.make(env_name)
        if use_wrapper and reward_config is not None:
            env = RocketRewardWrapper(env, reward_config)
        set_seed(int(seed))  # Ensure seed is Python int
        return env
    
    return factory

