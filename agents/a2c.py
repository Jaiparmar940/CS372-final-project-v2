"""
Actor-Critic (A2C) agent.
Implements Advantage Actor-Critic with separate policy and value networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
import os

from networks.policy_network import PolicyNetwork, create_policy_network
from networks.value_network import ValueNetwork, create_value_network
from utils.device import get_device, to_device
from utils.config import NetworkConfig, OptimizerConfig


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.
    Uses separate policy (actor) and value (critic) networks.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_config: Optional[NetworkConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_steps: int = 5,  # Number of steps before update
        gradient_clip: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize A2C agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            network_config: Network configuration
            optimizer_config: Optimizer configuration
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            n_steps: Number of steps before update (for n-step returns)
            gradient_clip: Gradient clipping threshold
            device: Device to use (CPU or CUDA)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_steps = n_steps
        self.gradient_clip = gradient_clip
        
        # Device
        self.device = device if device is not None else get_device()
        
        # Networks
        network_config = network_config or NetworkConfig()
        self.policy_network = create_policy_network(state_dim, action_dim, network_config).to(self.device)
        self.value_network = create_value_network(state_dim, network_config).to(self.device)
        
        # Optimizers (can use same or different optimizers for actor and critic)
        optimizer_config = optimizer_config or OptimizerConfig()
        self.policy_optimizer = self._create_optimizer(optimizer_config, self.policy_network)
        self.value_optimizer = self._create_optimizer(optimizer_config, self.value_network)
        
        # Learning rate schedulers
        self.policy_scheduler = None
        self.value_scheduler = None
        if optimizer_config.use_scheduler:
            self.policy_scheduler = self._create_scheduler(optimizer_config, self.policy_optimizer)
            self.value_scheduler = self._create_scheduler(optimizer_config, self.value_optimizer)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_entropies = []
        self.episode_values = []
        self.episode_dones = []
        
        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        # For checkpointing
        self.optimizer_config = optimizer_config
    
    def _create_optimizer(self, config: OptimizerConfig, network: nn.Module) -> optim.Optimizer:
        """
        Create optimizer for network.
        
        Args:
            config: Optimizer configuration
            network: Network to optimize
            
        Returns:
            Optimizer instance
        """
        if config.optimizer.lower() == "adam":
            return optim.Adam(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            return optim.SGD(
                network.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_scheduler(self, config: OptimizerConfig, optimizer: optim.Optimizer):
        """
        Create learning rate scheduler.
        
        Args:
            config: Optimizer configuration
            optimizer: Optimizer to schedule
            
        Returns:
            Learning rate scheduler or None
        """
        if config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
        elif config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                verbose=True
            )
        else:
            return None
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action from policy and estimate value.
        
        Args:
            state: Current state
            deterministic: If True, take argmax action (for evaluation)
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Entropy of policy distribution
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from policy
        dist = self.policy_network.forward(state_tensor)
        if deterministic:
            action = torch.argmax(dist.probs).item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        # Get entropy and value estimate (both don't need gradients during action selection)
        with torch.no_grad():
            entropy = dist.entropy()
            value = self.value_network(state_tensor)
        
        return action, log_prob, entropy, value
    
    def store_transition(self, state, action, reward, log_prob, entropy, value, done):
        """
        Store transition for episode.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            entropy: Entropy of policy
            value: Estimated state value
            done: Whether episode terminated
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
        self.episode_entropies.append(entropy)
        self.episode_values.append(value)
        self.episode_dones.append(done)
    
    def compute_advantages(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute advantages and returns using a bootstrap from next_value.
        
        Args:
            next_value: Value of next state (for bootstrapping after the rollout)
            
        Returns:
            advantages: List of advantages
            returns: List of returns
        """
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        dones = torch.FloatTensor(self.episode_dones).to(self.device)
        
        returns_tensor = torch.zeros_like(rewards).to(self.device)
        running_return = torch.tensor(next_value).to(self.device)
        
        # Compute discounted returns backward so each step bootstraps correctly
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns_tensor[t] = running_return
        
        advantages_tensor = returns_tensor - torch.stack(self.episode_values).squeeze().to(self.device)
        
        return advantages_tensor.tolist(), returns_tensor.tolist()
    
    def update(self, next_value: float = 0.0) -> Tuple[float, float, float]:
        """
        Update policy and value networks.
        
        Args:
            next_value: Value of next state (for bootstrapping)
            
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy_loss: Entropy loss value
        """
        if len(self.episode_states) == 0:
            return 0.0, 0.0, 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(next_value)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages (with numerical stability)
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        if adv_std > 1e-8:
            advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
        else:
            advantages_tensor = advantages_tensor - adv_mean  # Just center if std is too small
        
        # Check for NaN in advantages
        if torch.isnan(advantages_tensor).any():
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions_tensor = torch.LongTensor(self.episode_actions).to(self.device)
        
        # CRITICAL: Recompute log_probs with gradients enabled using CURRENT policy state
        # The stored log_probs are from action selection and may be detached
        dist = self.policy_network.forward(states_tensor)
        log_probs_tensor = dist.log_prob(actions_tensor)
        entropies_tensor = dist.entropy()
        
        # Check for NaN in log_probs
        if torch.isnan(log_probs_tensor).any():
            print("Warning: NaN detected in log_probs, replacing with zeros")
            log_probs_tensor = torch.nan_to_num(log_probs_tensor, nan=0.0)
        
        # Recompute values with gradients (needed for value loss)
        values_tensor = self.value_network(states_tensor)
        
        # Ensure proper shape (handle both [batch, 1] and [batch] cases)
        if values_tensor.dim() > 1:
            values_tensor = values_tensor.squeeze()
        if values_tensor.dim() == 0:
            values_tensor = values_tensor.unsqueeze(0)
        
        # Ensure values_tensor and returns_tensor have same shape
        if values_tensor.shape != returns_tensor.shape:
            values_tensor = values_tensor.view_as(returns_tensor)
        
        # Check for NaN in values
        if torch.isnan(values_tensor).any():
            print("Warning: NaN detected in values, replacing with zeros")
            values_tensor = torch.nan_to_num(values_tensor, nan=0.0)
        
        # Compute policy loss (actor) - use recomputed log_probs with gradients
        policy_loss = -(log_probs_tensor * advantages_tensor).mean()
        
        # Compute value loss (critic) - now values_tensor has gradients
        value_loss = nn.MSELoss()(values_tensor, returns_tensor)
        
        # Entropy regularization
        entropy_loss = -self.entropy_coef * entropies_tensor.mean()
        
        # Total losses
        total_policy_loss = policy_loss + entropy_loss
        total_value_loss = self.value_coef * value_loss
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        
        # Check gradient norm before clipping (for debugging)
        policy_grad_norm_before = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), float('inf'))
        if self.gradient_clip > 0:
            policy_grad_norm_after = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.gradient_clip)
        else:
            policy_grad_norm_after = policy_grad_norm_before
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        
        # Check gradient norm before clipping (for debugging)
        value_grad_norm_before = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), float('inf'))
        if self.gradient_clip > 0:
            value_grad_norm_after = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.gradient_clip)
        else:
            value_grad_norm_after = value_grad_norm_before
        self.value_optimizer.step()
        
        # Update learning rate schedulers
        if self.policy_scheduler is not None and isinstance(self.policy_scheduler, optim.lr_scheduler.StepLR):
            self.policy_scheduler.step()
        if self.value_scheduler is not None and isinstance(self.value_scheduler, optim.lr_scheduler.StepLR):
            self.value_scheduler.step()
        
        policy_loss_value = policy_loss.item()
        value_loss_value = value_loss.item()
        entropy_loss_value = entropy_loss.item()
        
        # Debug: Print gradient norms for first few updates to verify learning
        if len(self.policy_losses) < 3:
            print(f"Update {len(self.policy_losses)+1}: Policy grad norm: {policy_grad_norm_before:.4f} -> {policy_grad_norm_after:.4f}, "
                  f"Value grad norm: {value_grad_norm_before:.4f} -> {value_grad_norm_after:.4f}, "
                  f"Policy loss: {policy_loss_value:.4f}, Value loss: {value_loss_value:.4f}")
        
        self.policy_losses.append(policy_loss_value)
        self.value_losses.append(value_loss_value)
        self.entropy_losses.append(entropy_loss_value)
        
        # Clear episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_entropies = []
        self.episode_values = []
        self.episode_dones = []
        
        return policy_loss_value, value_loss_value, entropy_loss_value
    
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
        next_value = 0.0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, entropy, value = self.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            # Get next state value for bootstrapping
            if not (terminated or truncated):
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_value = self.value_network(next_state_tensor).item()
            else:
                next_value = 0.0
                episode_info = step_info
            
            # Store transition
            self.store_transition(
                state, action, reward, log_prob, entropy, value,
                terminated or truncated
            )
            
            total_reward += reward
            steps += 1
            
            # Update periodically (n-step)
            if len(self.episode_states) >= self.n_steps:
                self.update(next_value)
            
            if terminated or truncated:
                # Final update with remaining steps (use next_value=0 for terminal state)
                if len(self.episode_states) > 0:
                    self.update(0.0)
                break
            
            state = next_state
        
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
                action, _, _, _ = self.select_action(state, deterministic=True)
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
        Update learning rate schedulers with validation metric.
        
        Args:
            metric: Validation metric (e.g., mean return)
        """
        if self.policy_scheduler is not None and isinstance(self.policy_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.policy_scheduler.step(metric)
        if self.value_scheduler is not None and isinstance(self.value_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.value_scheduler.step(metric)
    
    def save(self, filepath: str):
        """Save agent to file."""
        checkpoint = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "value_network_state_dict": self.value_network.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
            "optimizer_config": self.optimizer_config
        }
        
        if self.policy_scheduler is not None:
            checkpoint["policy_scheduler_state_dict"] = self.policy_scheduler.state_dict()
        if self.value_scheduler is not None:
            checkpoint["value_scheduler_state_dict"] = self.value_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Saved A2C agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.episode_returns = checkpoint.get("episode_returns", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.entropy_losses = checkpoint.get("entropy_losses", [])
        
        if self.policy_scheduler is not None and "policy_scheduler_state_dict" in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
        if self.value_scheduler is not None and "value_scheduler_state_dict" in checkpoint:
            self.value_scheduler.load_state_dict(checkpoint["value_scheduler_state_dict"])
        
        print(f"Loaded A2C agent from {filepath}")

