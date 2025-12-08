# Attribution

This document credits external resources, libraries, and tools used in this project.

## External Libraries and Frameworks

### PyTorch
- **Purpose**: Deep learning framework for neural networks
- **License**: BSD-style license
- **Website**: https://pytorch.org/
- **Usage**: All neural network implementations (DQN, A2C)

### Gymnasium
- **Purpose**: Standard RL environment API
- **License**: MIT License
- **Website**: https://gymnasium.farama.org/
- **Usage**: LunarLander-v3 environment
- **Note**: Gymnasium is the maintained fork of OpenAI Gym

### NumPy
- **Purpose**: Numerical computing
- **License**: BSD License
- **Website**: https://numpy.org/
- **Usage**: Array operations and numerical computations

### Matplotlib
- **Purpose**: Plotting and visualization
- **License**: Matplotlib License
- **Website**: https://matplotlib.org/
- **Usage**: Learning curves and comparison plots

### Pandas
- **Purpose**: Data analysis and CSV handling
- **License**: BSD License
- **Website**: https://pandas.pydata.org/
- **Usage**: Hyperparameter sweep results and evaluation metrics

## Algorithms and References

### Q-Learning
- **Reference**: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
- **Implementation**: Deep Q-Network (DQN) implementation in `agents/dqn.py` uses Q-learning principles with neural network function approximation

### Deep Q-Network (DQN)
- **Reference**: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
- **Implementation**: Custom implementation in `agents/dqn.py` with experience replay and target network

### Actor-Critic (A2C)
- **Reference**: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- **Implementation**: Custom implementation in `agents/a2c.py`

## Environment

### LunarLander-v3
- **Source**: Gymnasium (Farama Foundation)
- **Description**: Continuous state space, discrete action space rocket landing simulation
- **Usage**: Main environment for deep RL agents
- **Note**: Requires Box2D dependencies (included in requirements.txt)

## AI Tools and Assistance

### Cursor AI / Claude
- **Purpose**: Code generation and assistance
- **Usage**: This project was developed with assistance from AI coding tools
- **Note**: All code was reviewed and adapted for the specific requirements of this project

## Code Structure and Design

The project structure and design patterns are inspired by:
- Standard RL library organization (e.g., Stable-Baselines3, Ray RLlib)
- PyTorch best practices for RL implementations
- Common patterns in academic RL research codebases

## Educational Resources

The following resources were consulted for algorithm understanding:
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd edition)
- Various online tutorials and documentation for PyTorch and Gymnasium

## License Note

This project is for educational purposes as part of a course final project (CS372). All external libraries are used in accordance with their respective licenses.

## Acknowledgments

- Farama Foundation for maintaining Gymnasium
- PyTorch team for the excellent deep learning framework
- The reinforcement learning research community for algorithm development and documentation

