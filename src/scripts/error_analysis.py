# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Create an error analysis script that analyzes failure cases and identifies when and why RL agents fail to land successfully"

"""
Error analysis: Analyze failure cases and agent behavior.
Identifies when and why agents fail to land successfully.
"""

import os
import sys
# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from environments.reward_wrapper import RocketRewardWrapper
from training.trainer import set_seed, create_env_factory
from utils.config import RewardConfig
from utils.device import get_device
import pandas as pd


def load_agent(algorithm: str, checkpoint_path: str):
    """Load trained agent from checkpoint."""
    import torch
    from utils.config import OptimizerConfig, NetworkConfig
    
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
    elif algorithm == "a2c":
        # Load checkpoint to get optimizer config first (to match optimizer type)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        optimizer_config = checkpoint.get("optimizer_config", OptimizerConfig())
        
        agent = A2CAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=optimizer_config,
            device=device
        )
        agent.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def analyze_failures(agent, env_factory, num_episodes: int = 50):
    """Analyze failure cases and collect statistics."""
    
    failure_cases = {
        "crashes": [],
        "timeouts": [],
        "bad_landings": []
    }
    
    state_statistics = {
        "crash_states": [],
        "success_states": [],
        "final_velocities": [],
        "final_positions": []
    }
    
    episode_outcomes = {
        "success": 0,
        "crash": 0,
        "timeout": 0
    }
    
    for episode in range(num_episodes):
        env = env_factory(seed=200 + episode)  # Use test seeds
        state, _ = env.reset()
        episode_states = [state.copy()]
        episode_rewards = []
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Handle different agent types
            if hasattr(agent, 'select_action'):
                # Check if it's DQN (takes training parameter) or A2C (returns tuple)
                try:
                    action_result = agent.select_action(state, training=False)
                    # A2C returns (action, log_prob, value, entropy), DQN returns int
                    action = action_result[0] if isinstance(action_result, tuple) else action_result
                except TypeError:
                    # A2C doesn't accept training parameter
                    action_result = agent.select_action(state)
                    action = action_result[0] if isinstance(action_result, tuple) else action_result
            else:
                raise ValueError("Agent does not have select_action method")
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_states.append(next_state.copy())
            episode_rewards.append(reward)
            
            state = next_state
            done = terminated or truncated
            steps += 1
        
        # Analyze outcome
        if done:
            if "crashed" in info and info["crashed"]:
                episode_outcomes["crash"] += 1
                failure_cases["crashes"].append({
                    "episode": episode,
                    "final_state": state,
                    "steps": steps,
                    "total_reward": sum(episode_rewards),
                    "final_velocity": np.sqrt(state[2]**2 + state[3]**2),  # Speed
                    "final_altitude": state[1],
                    "final_x": state[0]
                })
                state_statistics["crash_states"].append(state)
            elif "landed" in info and info["landed"]:
                episode_outcomes["success"] += 1
                state_statistics["success_states"].append(state)
            else:
                episode_outcomes["crash"] += 1
                failure_cases["bad_landings"].append({
                    "episode": episode,
                    "final_state": state,
                    "steps": steps,
                    "info": info
                })
        else:
            episode_outcomes["timeout"] += 1
            failure_cases["timeouts"].append({
                "episode": episode,
                "steps": steps,
                "final_state": state
            })
        
        # Collect final state statistics
        if len(state) >= 4:
            state_statistics["final_velocities"].append(np.sqrt(state[2]**2 + state[3]**2))
            state_statistics["final_positions"].append([state[0], state[1]])
        
        env.close()
    
    return {
        "failure_cases": failure_cases,
        "state_statistics": state_statistics,
        "episode_outcomes": episode_outcomes
    }


def visualize_error_analysis(analysis_results, agent_name: str, save_dir: str = "error_analysis"):
    """Create visualizations of error analysis."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    outcomes = analysis_results["episode_outcomes"]
    state_stats = analysis_results["state_statistics"]
    failures = analysis_results["failure_cases"]
    
    # 1. Outcome distribution pie chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pie chart of outcomes
    labels = list(outcomes.keys())
    sizes = [outcomes[k] for k in labels]
    colors = ['green', 'red', 'orange']
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title(f"{agent_name}: Episode Outcomes")
    
    # Crash analysis: final velocities
    if state_stats["crash_states"]:
        crash_velocities = [np.sqrt(s[2]**2 + s[3]**2) for s in state_stats["crash_states"]]
        success_velocities = [np.sqrt(s[2]**2 + s[3]**2) for s in state_stats["success_states"]] if state_stats["success_states"] else []
        
        axes[0, 1].hist(crash_velocities, bins=20, alpha=0.7, label="Crashes", color='red')
        if success_velocities:
            axes[0, 1].hist(success_velocities, bins=20, alpha=0.7, label="Successes", color='green')
        axes[0, 1].set_xlabel("Final Velocity (m/s)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Final Velocity Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Crash positions
    if failures["crashes"]:
        crash_x = [f["final_x"] for f in failures["crashes"]]
        crash_y = [f["final_altitude"] for f in failures["crashes"]]
        
        axes[1, 0].scatter(crash_x, crash_y, alpha=0.6, color='red', label="Crashes", s=50)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, label="Ground")
        axes[1, 0].axvline(x=0, color='blue', linestyle='--', linewidth=1, label="Landing Pad")
        axes[1, 0].set_xlabel("X Position")
        axes[1, 0].set_ylabel("Y Position (Altitude)")
        axes[1, 0].set_title("Crash Locations")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Failure reasons breakdown
    if failures["crashes"] or failures["timeouts"] or failures["bad_landings"]:
        failure_types = []
        failure_counts = []
        
        if failures["crashes"]:
            failure_types.append("Crashes")
            failure_counts.append(len(failures["crashes"]))
        if failures["timeouts"]:
            failure_types.append("Timeouts")
            failure_counts.append(len(failures["timeouts"]))
        if failures["bad_landings"]:
            failure_types.append("Bad Landings")
            failure_counts.append(len(failures["bad_landings"]))
        
        if failure_types:
            axes[1, 1].bar(failure_types, failure_counts, color=['red', 'orange', 'purple'], alpha=0.7)
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Failure Type Distribution")
            axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{agent_name}_error_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error analysis plot saved to: {save_path}")
    plt.close()


def run_error_analysis(algorithm: str, checkpoint_path: str, num_episodes: int = 50, use_reward_wrapper: bool = True):
    """Run error analysis for a trained agent."""
    
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS: {algorithm.upper()}")
    print(f"{'='*80}")
    
    # Load agent
    agent = load_agent(algorithm, checkpoint_path)
    # Set networks to evaluation mode (disables dropout, batch norm updates, etc.)
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    if hasattr(agent, 'target_network'):
        agent.target_network.eval()
    if hasattr(agent, 'policy_network'):
        agent.policy_network.eval()
    if hasattr(agent, 'value_network'):
        agent.value_network.eval()
    
    # Create environment factory
    reward_config = RewardConfig() if use_reward_wrapper else None
    def env_factory(seed):
        env = gym.make("LunarLander-v3")
        if use_reward_wrapper and reward_config:
            env = RocketRewardWrapper(env, reward_config)
        set_seed(seed)
        return env
    
    # Run analysis
    analysis = analyze_failures(agent, env_factory, num_episodes)
    
    # Print summary
    print("\nEpisode Outcomes:")
    for outcome, count in analysis["episode_outcomes"].items():
        percentage = (count / num_episodes) * 100
        print(f"  {outcome.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Analyze crash characteristics
    if analysis["failure_cases"]["crashes"]:
        crashes = analysis["failure_cases"]["crashes"]
        avg_crash_velocity = np.mean([c["final_velocity"] for c in crashes])
        avg_crash_altitude = np.mean([c["final_altitude"] for c in crashes])
        
        print(f"\nCrash Analysis ({len(crashes)} crashes):")
        print(f"  Average crash velocity: {avg_crash_velocity:.2f} m/s")
        print(f"  Average crash altitude: {avg_crash_altitude:.2f}")
        print(f"  Average crash X position: {np.mean([c['final_x'] for c in crashes]):.2f}")
    
    # Extract optimizer from checkpoint path (e.g., "dqn_adam_best.pt" -> "adam")
    checkpoint_name = os.path.basename(checkpoint_path)
    optimizer = None
    for opt in ["adam", "rmsprop", "sgd"]:
        if opt in checkpoint_name.lower():
            optimizer = opt
            break
    
    # Create agent name with optimizer
    if optimizer:
        agent_name = f"{algorithm}_{optimizer}"
    else:
        agent_name = algorithm
    
    # Create visualizations
    visualize_error_analysis(analysis, agent_name)
    
    # Save detailed results
    os.makedirs("error_analysis", exist_ok=True)
    
    if analysis["failure_cases"]["crashes"]:
        crash_df = pd.DataFrame(analysis["failure_cases"]["crashes"])
        crash_filename = f"{agent_name}_crash_details.csv"
        crash_df.to_csv(f"error_analysis/{crash_filename}", index=False)
        print(f"\nCrash details saved to: error_analysis/{crash_filename}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run error analysis on trained agent")
    parser.add_argument("--algorithm", type=str, required=True, choices=["dqn", "a2c"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to analyze")
    parser.add_argument("--use_reward_wrapper", action="store_true", default=True, help="Use reward wrapper (default: True)")
    parser.add_argument("--no_reward_wrapper", dest="use_reward_wrapper", action="store_false", help="Disable reward wrapper")
    
    args = parser.parse_args()
    
    set_seed(42)
    analysis = run_error_analysis(args.algorithm, args.checkpoint, args.num_episodes, args.use_reward_wrapper)
    print("\nError analysis complete!")

