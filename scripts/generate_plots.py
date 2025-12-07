"""
Regenerate all plots from saved agent checkpoints.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.plotting import plot_learning_curve, plot_comparison
from utils.device import get_device


def load_and_plot_agent(checkpoint_path, agent_type, plot_dir="plots"):
    """Load agent from checkpoint and generate learning curve plot."""
    device = get_device()
    
    if agent_type == "dqn":
        from agents.dqn import DQNAgent
        from utils.config import NetworkConfig, OptimizerConfig
        
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        
        if agent.episode_returns:
            plot_learning_curve(
                agent.episode_returns,
                title="DQN Learning Curve",
                save_path=os.path.join(plot_dir, "dqn_learning_curve.png")
            )
    
    elif agent_type == "reinforce":
        from agents.reinforce import REINFORCEAgent
        from utils.config import NetworkConfig, OptimizerConfig
        
        agent = REINFORCEAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        
        if agent.episode_returns:
            plot_learning_curve(
                agent.episode_returns,
                title="REINFORCE Learning Curve",
                save_path=os.path.join(plot_dir, "reinforce_learning_curve.png")
            )
    
    elif agent_type == "a2c":
        from agents.a2c import A2CAgent
        from utils.config import NetworkConfig, OptimizerConfig
        
        agent = A2CAgent(
            state_dim=8,
            action_dim=4,
            network_config=NetworkConfig(),
            optimizer_config=OptimizerConfig(),
            device=device
        )
        agent.load(checkpoint_path)
        
        if agent.episode_returns:
            plot_learning_curve(
                agent.episode_returns,
                title="A2C Learning Curve",
                save_path=os.path.join(plot_dir, "a2c_learning_curve.png")
            )


def main():
    print("Regenerating plots from checkpoints...")
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Check for checkpoints and generate plots
    checkpoint_dirs = {
        "dqn": "checkpoints/dqn",
        "reinforce": "checkpoints/reinforce",
        "a2c": "checkpoints/a2c"
    }
    
    for agent_type, checkpoint_dir in checkpoint_dirs.items():
        if os.path.exists(checkpoint_dir):
            best_checkpoint = os.path.join(checkpoint_dir, f"{agent_type}_best.pt")
            if os.path.exists(best_checkpoint):
                print(f"Loading {agent_type} from {best_checkpoint}...")
                load_and_plot_agent(best_checkpoint, agent_type, plot_dir)
    
    print("Plot generation complete!")


if __name__ == "__main__":
    main()

