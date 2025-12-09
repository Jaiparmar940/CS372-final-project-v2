# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create a visual testing script that runs RL agents in LunarLander with rendering enabled to visualize landing behavior"

"""
Visual testing script for RL agents in LunarLander environment.
Runs agent with rendering enabled to see the landing behavior.
"""

import argparse
import os
import sys
import time

# Add project root to path
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
from training.trainer import set_seed, create_env_factory
from environments.reward_wrapper import RocketRewardWrapper
from utils.config import RewardConfig
from utils.device import get_device


def load_agent(algorithm: str, checkpoint_path: str):
    """
    Load agent from checkpoint.
    
    Args:
        algorithm: Algorithm name ("dqn", "a2c")
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded agent
    """
    from agents.dqn import DQNAgent
    from agents.a2c import A2CAgent
    from utils.config import NetworkConfig, OptimizerConfig
    
    device = get_device()
    
    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        # Use v2 (may show deprecation warning but should still work)
        # Note: v3 requires Box2D which may not be installed
        return agent, "LunarLander-v3"
    
    elif algorithm == "a2c":
        agent = A2CAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        return agent, "LunarLander-v3"
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def test_visual(
    algorithm: str,
    checkpoint_path: str,
    num_episodes: int = 5,
    use_reward_wrapper: bool = True,
    render_mode: str = "human",
    delay: float = 0.01
):
    """
    Test agent visually in LunarLander environment.
    
    Args:
        algorithm: Algorithm name
        checkpoint_path: Path to checkpoint
        num_episodes: Number of episodes to run
        use_reward_wrapper: Whether to use reward wrapper
        render_mode: Rendering mode ("human", "rgb_array", or None)
        delay: Delay between frames (seconds)
    """
    print("="*80)
    print("VISUAL TESTING - LunarLander Environment")
    print("="*80)
    print(f"Algorithm: {algorithm}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Render mode: {render_mode}")
    print(f"Number of episodes: {num_episodes}")
    print()
    
    # Load agent
    try:
        agent, env_name = load_agent(algorithm, checkpoint_path)
        if agent is None:
            return
        print(f"Agent loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create environment with rendering
    reward_config = RewardConfig() if use_reward_wrapper else None
    
    def create_env(seed=None):
        try:
            # Try to create environment (may show deprecation warning for v2)
            env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("\nNote: LunarLander requires Box2D.")
            print("Install with: pip install swig && pip install 'gymnasium[box2d]'")
            raise
        if use_reward_wrapper and reward_config:
            env = RocketRewardWrapper(env, reward_config)
        if seed is not None:
            set_seed(seed)
        return env
    
    # Test episodes
    episode_returns = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        env = create_env(seed=200 + episode)  # Use test seeds
        state, info = env.reset()
        
        total_reward = 0.0
        steps = 0
        episode_info = {}
        
        while steps < 1000:  # Max steps
            # Select action (no exploration in eval mode)
            if algorithm == "dqn":
                action = agent.select_action(state, training=False)
            elif algorithm == "a2c":
                action, _, _, _ = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Render (if human mode, this displays the window)
            if render_mode == "human":
                env.render()
                if delay > 0:
                    time.sleep(delay)
            
            if terminated or truncated:
                episode_info = step_info
                break
            
            state = next_state
        
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        
        # Check outcome
        if episode_info.get("landed", False):
            success_count += 1
            outcome = "✓ SUCCESSFUL LANDING"
        elif episode_info.get("crashed", False):
            crash_count += 1
            outcome = "✗ CRASHED"
        else:
            outcome = "⚠ TIMEOUT"
        
        fuel_used = episode_info.get("fuel_used", 0)
        
        print(f"  Return: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Outcome: {outcome}")
        if fuel_used > 0:
            print(f"  Fuel used: {fuel_used:.2f}")
        
        env.close()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mean return: {sum(episode_returns) / len(episode_returns):.2f}")
    print(f"Std return: {(sum((r - sum(episode_returns)/len(episode_returns))**2 for r in episode_returns) / len(episode_returns))**0.5:.2f}")
    print(f"Mean episode length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Crash rate: {crash_count}/{num_episodes} ({crash_count/num_episodes*100:.1f}%)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Test agent visually in LunarLander")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["dqn", "a2c"],
                       help="Algorithm name")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--render_mode", type=str, default="human",
                       choices=["human", "rgb_array", "none"],
                       help="Rendering mode")
    parser.add_argument("--delay", type=float, default=0.01,
                       help="Delay between frames (seconds)")
    parser.add_argument("--no_wrapper", action="store_true",
                       help="Don't use reward wrapper")
    
    args = parser.parse_args()
    
    # Convert "none" to None for render_mode
    render_mode = None if args.render_mode == "none" else args.render_mode
    
    test_visual(
        algorithm=args.algorithm,
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        use_reward_wrapper=not args.no_wrapper,
        render_mode=render_mode,
        delay=args.delay
    )


if __name__ == "__main__":
    main()

