# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Implement a custom reward wrapper for LunarLander-v3 that parameterizes rewards for landing success, fuel usage, and smoothness of control"

"""
Custom reward wrapper for LunarLander-v3 environment.
Implements parameterized reward function that trades off landing success,
fuel usage, and smoothness of control.
"""

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from typing import Dict, Any, Optional
from utils.config import RewardConfig


class RocketRewardWrapper(Wrapper):
    """
    Wrapper that modifies LunarLander-v3 rewards to emphasize:
    - Safe landings
    - Fuel efficiency
    - Smooth control
    
    The reward function is parameterized for hyperparameter tuning.
    """
    
    def __init__(
        self,
        env: gym.Env,
        reward_config: Optional[RewardConfig] = None,
        track_fuel: bool = True
    ):
        """
        Initialize reward wrapper.
        
        Args:
            env: Gymnasium environment (LunarLander-v3)
            reward_config: Configuration for reward parameters
            track_fuel: Whether to track fuel consumption
        """
        super().__init__(env)
        self.reward_config = reward_config or RewardConfig()
        self.track_fuel = track_fuel
        
        # Track previous action for smoothness penalty
        self.prev_action = None
        
        # Track fuel usage
        self.total_fuel_used = 0.0
        self.episode_fuel = 0.0
        
        # Track episode statistics
        self.episode_stats = {
            "base_reward": 0.0,
            "fuel_penalty": 0.0,
            "smoothness_penalty": 0.0,
            "landing_bonus": 0.0,
            "crash_penalty": 0.0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_action = None
        self.episode_fuel = 0.0
        self.episode_stats = {
            "base_reward": 0.0,
            "fuel_penalty": 0.0,
            "smoothness_penalty": 0.0,
            "landing_bonus": 0.0,
            "crash_penalty": 0.0
        }
        return obs, info
    
    def step(self, action):
        """
        Execute step with custom reward function.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Scale base reward
        modified_reward = base_reward * self.reward_config.base_reward_scale
        self.episode_stats["base_reward"] += base_reward
        
        # Compute fuel consumption penalty
        # In LunarLander, actions 1, 2, 3 use engines (fuel)
        fuel_used = 0.0
        if action in [1, 2, 3]:
            # Main engine (action 1) uses more fuel
            if action == 1:
                fuel_used = 1.0
            else:
                fuel_used = 0.5  # Side engines use less fuel
        
        fuel_penalty = -self.reward_config.fuel_penalty * fuel_used
        modified_reward += fuel_penalty
        self.episode_fuel += fuel_used
        self.episode_stats["fuel_penalty"] += fuel_penalty
        
        # Smoothness penalty: penalize large changes in actions
        if self.prev_action is not None:
            action_change = abs(action - self.prev_action)
            if action_change > 0:
                smoothness_penalty = -self.reward_config.smoothness_penalty * action_change
                modified_reward += smoothness_penalty
                self.episode_stats["smoothness_penalty"] += smoothness_penalty
        
        self.prev_action = action
        
        # Landing and crash bonuses/penalties
        if terminated:
            # Check if landed successfully
            # In LunarLander, we can check if the lander is on the pad
            # This is approximated by checking if episode ended and reward was positive
            if base_reward > 0:
                # Successful landing
                landing_bonus = self.reward_config.landing_bonus
                modified_reward += landing_bonus
                self.episode_stats["landing_bonus"] = landing_bonus
                info["landed"] = True
                info["crashed"] = False
            else:
                # Crash
                crash_penalty = self.reward_config.crash_penalty
                modified_reward += crash_penalty
                self.episode_stats["crash_penalty"] = crash_penalty
                info["landed"] = False
                info["crashed"] = True
        
        # Add fuel usage to info
        if self.track_fuel:
            info["fuel_used"] = self.episode_fuel
            info["total_fuel_used"] = self.total_fuel_used + self.episode_fuel
        
        # Add reward breakdown to info
        info["reward_breakdown"] = self.episode_stats.copy()
        
        return obs, modified_reward, terminated, truncated, info
    
    def get_fuel_usage(self) -> float:
        """Get total fuel used in current episode."""
        return self.episode_fuel
    
    def get_reward_stats(self) -> Dict[str, float]:
        """Get episode reward statistics."""
        return self.episode_stats.copy()
    
    def update_reward_config(self, reward_config: RewardConfig):
        """Update reward configuration (for hyperparameter tuning)."""
        self.reward_config = reward_config

