# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Implement configuration management with dataclasses for network config, optimizer config, training config, and reward config"

"""
Configuration management for RL experiments.
Supports YAML/JSON config files and easy parameter switching.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os


@dataclass
class BaseConfig:
    """Base configuration class for RL experiments."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class RewardConfig:
    """Configuration for custom reward function."""
    landing_bonus: float = 100.0
    fuel_penalty: float = 0.1
    crash_penalty: float = -100.0
    smoothness_penalty: float = 0.05
    base_reward_scale: float = 1.0


@dataclass
class TrainingConfig:
    """Base training configuration."""
    # Seeds for train/validation/test splits
    train_seeds: list = None
    val_seeds: list = None
    test_seeds: list = None
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    gamma: float = 0.99
    
    # Validation parameters
    val_frequency: int = 20  # Validate every N episodes
    val_episodes_per_seed: int = 5  # Number of episodes per seed during validation
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.01
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_frequency: int = 100
    
    # Logging
    log_dir: str = "data/logs"
    log_frequency: int = 10
    
    def __post_init__(self):
        if self.train_seeds is None:
            self.train_seeds = list(range(42, 52))
        if self.val_seeds is None:
            self.val_seeds = list(range(100, 110))
        if self.test_seeds is None:
            self.test_seeds = list(range(200, 210))


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    hidden_sizes: list = None
    dropout_rate: float = 0.0
    use_dropout: bool = False
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 128]


@dataclass
class OptimizerConfig:
    """Configuration for optimizers and learning rate scheduling."""
    optimizer: str = "adam"  # "adam", "rmsprop", "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0  # L2 regularization
    momentum: float = 0.9  # For SGD
    
    # Learning rate scheduling
    use_scheduler: bool = False
    scheduler_type: str = "step"  # "step", "plateau"
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.9
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config_from_json(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

