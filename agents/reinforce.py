"""
REINFORCE (Monte Carlo Policy Gradient) agent.
Implements vanilla policy gradient with entropy regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
import os

from networks.policy_network import PolicyNetwork, create_policy_network
from utils.device import get_device, to_device
from utils.config import NetworkConfig, OptimizerConfig


class REINFORCEAgent:
    """
    REINFORCE agent using Monte Carlo policy gradient.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_config: Optional[NetworkConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        use_baseline: bool = False,
        gradient_clip: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            network_config: Network configuration
            optimizer_config: Optimizer configuration
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            use_baseline: Whether to use baseline subtraction
            gradient_clip: Gradient clipping threshold
            device: Device to use (CPU or CUDA)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.gradient_clip = gradient_clip
        
        # Device
        self.device = device if device is not None else get_device()
        
        # Policy network
        network_config = network_config or NetworkConfig()
        self.policy_network = create_policy_network(state_dim, action_dim, network_config).to(self.device)
        
        # Optimizer
        optimizer_config = optimizer_config or OptimizerConfig()
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Learning rate scheduler
        self.scheduler = None
        if optimizer_config.use_scheduler:
            self.scheduler = self._create_scheduler(optimizer_config)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_entropies = []
        
        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.policy_losses = []
        self.entropy_losses = []
        
        # For checkpointing
        self.optimizer_config = optimizer_config
    
    def _create_optimizer(self, config: OptimizerConfig) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            config: Optimizer configuration
            
        Returns:
            Optimizer instance
        """
        if config.optimizer.lower() == "adam":
            return optim.Adam(
                self.policy_network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.policy_network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.policy_network.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_scheduler(self, config: OptimizerConfig):
        """
        Create learning rate scheduler.
        
        Args:
            config: Optimizer configuration
            
        Returns:
            Learning rate scheduler or None
        """
        if config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
        elif config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                verbose=True
            )
        else:
            return None
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Entropy of policy distribution
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy_network.get_action(state_tensor)
        
        # Get entropy
        with torch.no_grad():
            dist = self.policy_network.forward(state_tensor)
            entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def store_transition(self, state, action, reward, log_prob, entropy):
        """
        Store transition for episode.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            entropy: Entropy of policy
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
        self.episode_entropies.append(entropy)
    
    def compute_returns(self) -> List[float]:
        """
        Compute discounted returns for episode.
        
        Returns:
            List of returns for each step
        """
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self) -> Tuple[float, float]:
        """
        Update policy network using REINFORCE algorithm.
        
        Returns:
            policy_loss: Policy loss value
            entropy_loss: Entropy loss value
        """
        if len(self.episode_states) == 0:
            return 0.0, 0.0
        
        # Compute returns
        returns = self.compute_returns()
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns (baseline subtraction)
        if self.use_baseline:
            returns_tensor = returns_tensor - returns_tensor.mean()
        
        # Convert states and actions to tensors
        states_tensor = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions_tensor = torch.LongTensor(self.episode_actions).to(self.device)
        log_probs_tensor = torch.stack(self.episode_log_probs).to(self.device)
        entropies_tensor = torch.stack(self.episode_entropies).to(self.device)
        
        # Compute policy gradient loss
        # REINFORCE: ∇J = E[∇log π(a|s) * G]
        policy_loss = -(log_probs_tensor * returns_tensor).mean()
        
        # Entropy regularization
        entropy_loss = -self.entropy_coef * entropies_tensor.mean()
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update learning rate scheduler
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.StepLR):
            self.scheduler.step()
        
        policy_loss_value = policy_loss.item()
        entropy_loss_value = entropy_loss.item()
        
        self.policy_losses.append(policy_loss_value)
        self.entropy_losses.append(entropy_loss_value)
        
        # Clear episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_entropies = []
        
        return policy_loss_value, entropy_loss_value
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int, Dict]:
        """
        Train for one episode.
        
        Args:
            env: Environment to interact with
            max_steps: Maximum steps per episode
            
        Returns:
            episode_return: Total return for episode
            episode_length: Number of steps
            info: Additional episode information
        """
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        episode_info = {}
        
        for step in range(max_steps):
            # Select action
            action, log_prob, entropy = self.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, log_prob, entropy)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                episode_info = step_info
                break
            
            state = next_state
        
        # Update policy after episode
        policy_loss, entropy_loss = self.update()
        
        # Record statistics
        self.episode_returns.append(total_reward)
        self.episode_lengths.append(steps)
        
        return total_reward, steps, episode_info
    
    def evaluate(self, env, num_episodes: int = 10, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to run
            seed: Random seed for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        success_count = 0
        crash_count = 0
        fuel_usage = []
        
        for episode in range(num_episodes):
            if seed is not None:
                state, info = env.reset(seed=seed + episode)
            else:
                state, info = env.reset()
            
            total_reward = 0.0
            steps = 0
            episode_info = {}
            
            while steps < 1000:  # Max steps
                action, _, _ = self.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    episode_info = step_info
                    break
                
                state = next_state
            
            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_info.get("landed", False):
                success_count += 1
            elif episode_info.get("crashed", False):
                crash_count += 1
            
            if "fuel_used" in episode_info:
                fuel_usage.append(episode_info["fuel_used"])
        
        metrics = {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes,
            "crash_rate": crash_count / num_episodes
        }
        
        if fuel_usage:
            metrics["mean_fuel_usage"] = np.mean(fuel_usage)
            metrics["std_fuel_usage"] = np.std(fuel_usage)
        
        return metrics
    
    def update_scheduler(self, metric: float):
        """
        Update learning rate scheduler with validation metric.
        
        Args:
            metric: Validation metric (e.g., mean return)
        """
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
    
    def save(self, filepath: str):
        """Save agent to file."""
        checkpoint = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "entropy_losses": self.entropy_losses,
            "optimizer_config": self.optimizer_config
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Saved REINFORCE agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_returns = checkpoint.get("episode_returns", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.entropy_losses = checkpoint.get("entropy_losses", [])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded REINFORCE agent from {filepath}")

