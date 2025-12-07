"""
Policy network for policy gradient methods (REINFORCE, A2C).
Outputs action probabilities over discrete action space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from utils.config import NetworkConfig


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Used for REINFORCE and as the actor in A2C.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = "relu"
    ):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
        """
        super(PolicyNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Select activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        input_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        
        # Output layer: logits for each action
        layers.append(nn.Linear(input_size, action_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for layer in self.layers[:-1]:  # All layers except output
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer with smaller weights for stable policy
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.01)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Forward pass through network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Categorical distribution over actions
        """
        x = state
        
        # Pass through hidden layers
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        
        # Output layer: action logits
        logits = self.layers[-1](x)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        return dist
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action index
            log_prob: Log probability of the action
        """
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action under current policy.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of policy distribution
            dist: Policy distribution
        """
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, dist


def create_policy_network(
    state_dim: int,
    action_dim: int,
    config: Optional[NetworkConfig] = None
) -> PolicyNetwork:
    """
    Factory function to create policy network from config.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        config: Network configuration
        
    Returns:
        Policy network instance
    """
    if config is None:
        config = NetworkConfig()
    
    return PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation
    )

