# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Implement a custom DQN Q-network architecture with configurable hidden layers, dropout support, and ReLU activation"

"""
Custom DQN Q-network architecture.
Multi-layer neural network for approximating Q-values in continuous state spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from utils.config import NetworkConfig


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for approximating Q(s, a).
    
    Custom architecture with configurable hidden layers and dropout support.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        dropout_rate: float = 0.0,
        use_dropout: bool = False,
        activation: str = "relu"
    ):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability (0.0 to 1.0)
            use_dropout: Whether to use dropout layers
            activation: Activation function ("relu", "tanh", "elu")
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dropout = use_dropout and dropout_rate > 0.0
        self.dropout_rate = dropout_rate
        
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
        
        # Output layer: Q-values for each action
        layers.append(nn.Linear(input_size, action_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for layer in self.layers[:-1]:  # All layers except output
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.01)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        x = state
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Check for NaN and replace with zeros
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
            
            # Apply dropout if enabled
            if self.use_dropout:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Output layer (no activation, raw Q-values)
        q_values = self.layers[-1](x)
        
        # Check for NaN in output
        if torch.isnan(q_values).any():
            q_values = torch.nan_to_num(q_values, nan=0.0)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()


def create_dqn_network(
    state_dim: int,
    action_dim: int,
    config: Optional[NetworkConfig] = None
) -> DQNNetwork:
    """
    Factory function to create DQN network from config.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        config: Network configuration
        
    Returns:
        DQN network instance
    """
    if config is None:
        config = NetworkConfig()
    
    return DQNNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate,
        use_dropout=config.use_dropout,
        activation=config.activation
    )

