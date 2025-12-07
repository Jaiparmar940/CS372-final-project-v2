"""
Tabular Q-learning agent for discrete state-action spaces.
Implements epsilon-greedy exploration and tracks learning progress.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os


class TabularQLearning:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.
    
    Suitable for small, discrete state-action spaces like the toy rocket environment.
    """
    
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        epsilon_decay_type: str = "exponential"  # "exponential" or "linear"
    ):
        """
        Initialize tabular Q-learning agent.
        
        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon value
            epsilon_decay: Decay rate for epsilon
            epsilon_decay_type: Type of decay ("exponential" or "linear")
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_type = epsilon_decay_type
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((state_space_size, action_space_size))
        
        # Track learning progress
        self.episode_returns = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.q_value_changes = []  # Track Q-value stability for convergence
        
        # For convergence detection
        self.prev_q_table = None
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (integer index)
            training: Whether in training mode (uses epsilon-greedy if True)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: greedy action
            return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update Q-value using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Current Q-value
        current_q = self.Q[state, action]
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        # Update Q-value using Bellman equation
        self.Q[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self, episode: int = None):
        """
        Decay epsilon according to schedule.
        
        Args:
            episode: Current episode number (for linear decay)
        """
        if self.epsilon_decay_type == "exponential":
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.epsilon_decay_type == "linear":
            if episode is not None:
                # Linear decay over episodes
                total_decay = self.epsilon_start - self.epsilon_end
                # Assume decay over 1000 episodes by default
                decay_per_episode = total_decay / 1000.0
                self.epsilon = max(self.epsilon_end, self.epsilon_start - episode * decay_per_episode)
            else:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            raise ValueError(f"Unknown epsilon_decay_type: {self.epsilon_decay_type}")
    
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
        
        # Convert state to integer if needed
        if isinstance(state, np.ndarray):
            state = env.get_state_hash()
        
        total_reward = 0.0
        steps = 0
        episode_info = {"landed": False, "crashed": False}
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            # Convert next_state to integer if needed
            if isinstance(next_state, np.ndarray):
                next_state = env.get_state_hash()
            
            # Update Q-values
            self.update(state, action, reward, next_state, terminated or truncated)
            
            # Update tracking
            total_reward += reward
            steps += 1
            
            # Check for termination
            if terminated or truncated:
                episode_info.update(step_info)
                break
            
            state = next_state
        
        # Record episode statistics
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
        
        for episode in range(num_episodes):
            if seed is not None:
                state, info = env.reset(seed=seed + episode)
            else:
                state, info = env.reset()
            
            # Convert state to integer if needed
            if isinstance(state, np.ndarray):
                state = env.get_state_hash()
            
            total_reward = 0.0
            steps = 0
            episode_info = {}
            
            while steps < 1000:  # Max steps
                # Select greedy action (no exploration)
                action = self.select_action(state, training=False)
                
                # Take step
                next_state, reward, terminated, truncated, step_info = env.step(action)
                
                # Convert next_state to integer if needed
                if isinstance(next_state, np.ndarray):
                    next_state = env.get_state_hash()
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    episode_info = step_info
                    break
                
                state = next_state
            
            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            
            # Track success/crash
            if episode_info.get("landed", False):
                success_count += 1
            elif episode_info.get("crashed", False):
                crash_count += 1
        
        metrics = {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes,
            "crash_rate": crash_count / num_episodes
        }
        
        return metrics
    
    def check_convergence(self, threshold: float = 0.01) -> bool:
        """
        Check if Q-values have converged (changed less than threshold).
        
        Args:
            threshold: Maximum change in Q-values to consider converged
            
        Returns:
            True if converged, False otherwise
        """
        if self.prev_q_table is None:
            self.prev_q_table = self.Q.copy()
            return False
        
        max_change = np.max(np.abs(self.Q - self.prev_q_table))
        self.q_value_changes.append(max_change)
        self.prev_q_table = self.Q.copy()
        
        return max_change < threshold
    
    def save(self, filepath: str):
        """Save agent (Q-table and hyperparameters) to file."""
        save_dict = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "epsilon_history": self.epsilon_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.Q = save_dict["Q"]
        self.epsilon = save_dict["epsilon"]
        self.learning_rate = save_dict["learning_rate"]
        self.gamma = save_dict["gamma"]
        self.episode_returns = save_dict.get("episode_returns", [])
        self.episode_lengths = save_dict.get("episode_lengths", [])
        self.epsilon_history = save_dict.get("epsilon_history", [])
        print(f"Loaded agent from {filepath}")

