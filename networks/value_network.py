"""
Value network for actor-critic methods (A2C).
Estimates state values V(s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from utils.config import NetworkConfig


class ValueNetwork(nn.Module):
    """
    Value network that estimates V(s).
    
    Used as the critic in A2C.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: List[int] = None,
        dropout_rate: float = 0.0,
        use_dropout: bool = False,
        activation: str = "relu"
    ):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability (0.0 to 1.0)
            use_dropout: Whether to use dropout layers
            activation: Activation function ("relu", "tanh", "elu")
        """
        super(ValueNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        
        self.state_dim = state_dim
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
        
        # Output layer: single scalar value
        layers.append(nn.Linear(input_size, 1))
        
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
            State values, shape (batch_size, 1)
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
        
        # Output layer: scalar value
        value = self.layers[-1](x)
        
        # Check for NaN in output and clamp
        if torch.isnan(value).any():
            value = torch.nan_to_num(value, nan=0.0)
        value = torch.clamp(value, min=-100.0, max=100.0)
        
        return value


def create_value_network(
    state_dim: int,
    config: Optional[NetworkConfig] = None
) -> ValueNetwork:
    """
    Factory function to create value network from config.
    
    Args:
        state_dim: Dimension of state space
        config: Network configuration
        
    Returns:
        Value network instance
    """
    if config is None:
        config = NetworkConfig()
    
    return ValueNetwork(
        state_dim=state_dim,
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate,
        use_dropout=config.use_dropout,
        activation=config.activation
    )

