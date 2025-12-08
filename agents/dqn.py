"""
Deep Q-Network (DQN) agent with experience replay and target network.
Implements standard DQN algorithm with configurable optimizers and regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, Optional, List, Dict
import random
import os
import pickle

from networks.dqn_network import DQNNetwork, create_dqn_network
from utils.device import get_device, to_device
from utils.config import NetworkConfig, OptimizerConfig


class ExperienceReplay:
    """
    Experience replay buffer for DQN.
    Stores and samples transitions for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_config: Optional[NetworkConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_frequency: int = 100,
        target_update_type: str = "hard",  # "hard" or "soft"
        tau: float = 0.01,  # For soft updates
        gradient_clip: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            network_config: Network configuration
            optimizer_config: Optimizer configuration
            replay_buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon value
            epsilon_decay: Decay rate for epsilon
            target_update_frequency: Frequency of target network updates
            target_update_type: Type of update ("hard" or "soft")
            tau: Soft update coefficient (for soft updates)
            gradient_clip: Gradient clipping threshold
            device: Device to use (CPU or CUDA)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.target_update_type = target_update_type
        self.tau = tau
        self.gradient_clip = gradient_clip
        
        # Device
        self.device = device if device is not None else get_device()
        
        # Networks
        network_config = network_config or NetworkConfig()
        self.q_network = create_dqn_network(state_dim, action_dim, network_config).to(self.device)
        self.target_network = create_dqn_network(state_dim, action_dim, network_config).to(self.device)
        
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer
        optimizer_config = optimizer_config or OptimizerConfig()
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Learning rate scheduler
        self.scheduler = None
        if optimizer_config.use_scheduler:
            self.scheduler = self._create_scheduler(optimizer_config)
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(replay_buffer_size)
        
        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_history = []
        self.update_count = 0
        
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
                self.q_network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.q_network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.q_network.parameters(),
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
                mode='max',  # Maximize validation return
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                verbose=True
            )
        else:
            return None
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in experience replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Update Q-network using experience replay.
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (using numpy arrays first for better performance)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update learning rate scheduler
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.StepLR):
            self.scheduler.step()
        
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_frequency == 0:
            if self.target_update_type == "hard":
                self.target_network.load_state_dict(self.q_network.state_dict())
            elif self.target_update_type == "soft":
                # Soft update: θ_target = τ * θ_q + (1 - τ) * θ_target
                for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
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
            action = self.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, next_state, terminated or truncated)
            
            # Update Q-network
            loss = self.update()
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                episode_info = step_info
                break
            
            state = next_state
        
        # Record statistics
        self.episode_returns.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.epsilon)
        
        # Decay epsilon
        self.decay_epsilon()
        
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
                action = self.select_action(state, training=False)
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
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "epsilon_history": self.epsilon_history,
            "optimizer_config": self.optimizer_config
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Saved DQN agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.episode_returns = checkpoint.get("episode_returns", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.losses = checkpoint.get("losses", [])
        self.epsilon_history = checkpoint.get("epsilon_history", [])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded DQN agent from {filepath}")

