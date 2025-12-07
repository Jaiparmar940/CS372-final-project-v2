"""RL agents for rocket landing task."""

from .tabular_q_learning import TabularQLearning
from .dqn import DQNAgent
from .reinforce import REINFORCEAgent
from .a2c import A2CAgent

__all__ = ['TabularQLearning', 'DQNAgent', 'REINFORCEAgent', 'A2CAgent']

