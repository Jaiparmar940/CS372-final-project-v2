"""
Simple discrete tabular rocket landing environment.
Small finite state space suitable for tabular Q-learning.
"""

import numpy as np
from typing import Tuple, Optional
import gymnasium as gym
from gymnasium import spaces


class ToyRocketEnv(gym.Env):
    """
    Simple discrete grid-based rocket landing environment.
    
    State space: (x, y, vx, vy) where:
        - x, y: position on 10x10 grid (0-9)
        - vx, vy: velocity components (-1, 0, 1)
    
    Action space: 6 discrete actions
        0: no-op
        1: thrust up
        2: thrust down
        3: thrust left
        4: thrust right
        5: land (only valid when near landing pad)
    
    Goal: Land safely at position (5, 0) with low velocity.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, grid_size: int = 10, render_mode: Optional[str] = None):
        """
        Initialize toy rocket environment.
        
        Args:
            grid_size: Size of the grid (default 10x10)
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Landing pad is at (grid_size//2, 0)
        self.landing_pad_x = grid_size // 2
        self.landing_pad_y = 0
        
        # State: (x, y, vx, vy)
        # x, y: 0 to grid_size-1
        # vx, vy: -1, 0, or 1
        self.observation_space = spaces.MultiDiscrete([
            grid_size,  # x position
            grid_size,  # y position
            3,  # vx: -1, 0, 1 (mapped to 0, 1, 2)
            3   # vy: -1, 0, 1 (mapped to 0, 1, 2)
        ])
        
        # 6 actions: no-op, up, down, left, right, land
        self.action_space = spaces.Discrete(6)
        
        # Initialize state
        self.state = None
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation as array."""
        # Map velocities from -1,0,1 to 0,1,2 for MultiDiscrete
        vx_mapped = self.vx + 1
        vy_mapped = self.vy + 1
        return np.array([self.x, self.y, vx_mapped, vy_mapped], dtype=np.int32)
    
    def _velocity_from_mapped(self, v_mapped: int) -> int:
        """Convert mapped velocity (0,1,2) back to (-1,0,1)."""
        return v_mapped - 1
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Start at random position in upper half of grid
        self.x = self.np_random.integers(0, self.grid_size)
        self.y = self.np_random.integers(self.grid_size // 2, self.grid_size)
        
        # Start with random small velocity
        self.vx = self.np_random.integers(-1, 2)
        self.vy = self.np_random.integers(-1, 2)
        
        self.state = self._get_observation()
        
        info = {}
        return self.state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-5)
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended successfully
            truncated: Whether episode was truncated
            info: Additional info
        """
        # Update velocity based on action
        if action == 1:  # thrust up
            self.vy = max(-1, self.vy - 1)  # Reduce downward velocity
        elif action == 2:  # thrust down
            self.vy = min(1, self.vy + 1)  # Increase downward velocity
        elif action == 3:  # thrust left
            self.vx = max(-1, self.vx - 1)  # Reduce rightward velocity
        elif action == 4:  # thrust right
            self.vx = min(1, self.vx + 1)  # Increase rightward velocity
        elif action == 5:  # land
            # Only valid if near landing pad
            if abs(self.x - self.landing_pad_x) <= 1 and self.y <= 1:
                # Check if velocity is low enough for safe landing
                if abs(self.vx) <= 1 and abs(self.vy) <= 1:
                    # Successful landing (increased reward)
                    reward = 200.0  # Increased from 100 to make landing more attractive
                    terminated = True
                    truncated = False
                    info = {"landed": True, "crashed": False}
                    return self._get_observation(), reward, terminated, truncated, info
                else:
                    # Crash due to high velocity
                    reward = -50.0
                    terminated = True
                    truncated = False
                    info = {"landed": False, "crashed": True, "reason": "high_velocity"}
                    return self._get_observation(), reward, terminated, truncated, info
            else:
                # Invalid landing attempt
                reward = -10.0
                terminated = False
                truncated = False
                info = {"landed": False, "crashed": False}
        
        # Apply velocity to position
        new_x = self.x + self.vx
        new_y = self.y + self.vy
        
        # Apply gravity (always pulls down)
        self.vy = min(1, self.vy + 1)
        
        # Update position
        self.x = np.clip(new_x, 0, self.grid_size - 1)
        self.y = np.clip(new_y, 0, self.grid_size - 1)
        
        # Check boundaries
        if self.x <= 0 or self.x >= self.grid_size - 1:
            self.vx = 0  # Bounce or stop at boundary
        
        # Compute reward
        reward = -0.01  # Small step penalty (reduced to allow longer episodes)
        
        # Penalty for using thrust (fuel consumption)
        if action in [1, 2, 3, 4]:
            reward -= 0.1  # Reduced fuel penalty
        
        # Bonus for being near landing pad (increased to guide agent)
        distance_to_pad = abs(self.x - self.landing_pad_x) + abs(self.y - self.landing_pad_y)
        reward += 0.5 * (self.grid_size - distance_to_pad) / self.grid_size  # Increased proximity bonus
        
        # Additional bonus for being at low altitude (closer to landing)
        if self.y <= 2:
            reward += 0.2  # Bonus for being near ground
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Crash if hit ground without landing
        if self.y == 0 and action != 5:
            reward = -50.0
            terminated = True
            info = {"landed": False, "crashed": True, "reason": "ground_crash"}
        elif self.y == 0 and action == 5:
            # Attempted landing but conditions not met
            if abs(self.x - self.landing_pad_x) > 1 or abs(self.vx) > 1 or abs(self.vy) > 1:
                reward = -50.0
                terminated = True
                info = {"landed": False, "crashed": True, "reason": "bad_landing"}
            else:
                # Should have been caught above, but just in case
                reward = 200.0  # Increased from 100
                terminated = True
                info = {"landed": True, "crashed": False}
        else:
            info = {"landed": False, "crashed": False}
        
        self.state = self._get_observation()
        return self.state, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (simple text-based for now)."""
        if self.render_mode == "human":
            grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            
            # Mark landing pad
            grid[self.landing_pad_y][self.landing_pad_x] = 'L'
            
            # Mark rocket
            if 0 <= self.y < self.grid_size and 0 <= self.x < self.grid_size:
                grid[self.y][self.x] = 'R'
            
            print("\n" + "=" * (self.grid_size * 2 + 1))
            for row in reversed(grid):
                print("|" + " ".join(row) + "|")
            print("=" * (self.grid_size * 2 + 1))
            print(f"Position: ({self.x}, {self.y}), Velocity: ({self.vx}, {self.vy})")
    
    def get_state_hash(self) -> int:
        """
        Get hashable state representation for tabular Q-learning.
        
        Returns:
            Integer hash of current state
        """
        # Convert state to single integer for Q-table indexing
        # State: (x, y, vx, vy) where:
        #   x, y: 0 to grid_size-1
        #   vx, vy: -1, 0, 1 (mapped to 0, 1, 2)
        # Hash formula: x * (grid_size * 3 * 3) + y * (3 * 3) + (vx+1) * 3 + (vy+1)
        vx_mapped = self.vx + 1  # Maps -1,0,1 to 0,1,2
        vy_mapped = self.vy + 1  # Maps -1,0,1 to 0,1,2
        return int(self.x * (self.grid_size * 3 * 3) + 
                   self.y * (3 * 3) + 
                   vx_mapped * 3 + 
                   vy_mapped)
    
    def get_state_space_size(self) -> int:
        """Get total number of possible states."""
        return self.grid_size ** 2 * 3 * 3  # x, y, vx, vy

