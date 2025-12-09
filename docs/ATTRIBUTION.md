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

### PyYAML
- **Purpose**: YAML parser for configuration files
- **License**: MIT License
- **Website**: https://pyyaml.org/
- **Usage**: Configuration file parsing in `utils/config.py`

### Box2D-py
- **Purpose**: 2D physics engine for game development
- **License**: zlib License
- **Website**: https://github.com/pybox2d/pybox2d
- **Usage**: Required dependency for LunarLander-v3 environment physics simulation

### Pygame
- **Purpose**: Cross-platform multimedia library for game development
- **License**: LGPL License
- **Website**: https://www.pygame.org/
- **Usage**: Rendering and visualization for LunarLander-v3 environment

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
- **Usage**: This project was developed with assistance from AI coding tools. Each file modified or written by Cursor contains a note at the top stating its authors and prompting if relevant. All tuning was performed by Jay Parmar and Ryan Christ.
- **Note**: All code was reviewed and adapted for the specific requirements of this project

## AI-Generated Code Attribution

This project extensively used AI coding assistants (Cursor AI / Claude) for code generation and development. The following attribution practices are followed:

### File-Level Attribution

**Every Python file in this project that was created or significantly modified with AI assistance contains a header comment** at the top of the file with the following format:

```python
# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "[description of what the file does]"
```

**Example from `agents/dqn.py`:**
```python
# This file was created by Cursor.
# To recreate this file, prompt Cursor with: "Implement a Deep Q-Network (DQN) agent with experience replay buffer and target network"
```

### Attribution Details

1. **File Headers**: All AI-generated or AI-assisted files include attribution comments at the top indicating:
   - That the file was created by Cursor
   - The prompt that can be used to recreate similar functionality

2. **Code Review and Adaptation**: While AI tools were used for code generation, all code was:
   - Reviewed by team members (Jay Parmar and Ryan Christ)
   - Adapted to meet specific project requirements
   - Tested and validated for correctness
   - Integrated into the overall project architecture

3. **Human Contributions**: The following aspects were primarily human-directed:
   - Project architecture and design decisions
   - Hyperparameter tuning and optimization
   - Training and evaluation procedures
   - Documentation and analysis
   - Algorithm selection and implementation strategy

4. **Transparency**: This attribution approach ensures:
   - Full transparency about AI tool usage
   - Clear documentation of which components were AI-assisted
   - Reproducibility through documented prompts
   - Compliance with academic integrity standards

### Files with AI Attribution

The following categories of files contain AI attribution headers:
- **Agent implementations** (`agents/`): DQN, A2C implementations
- **Network architectures** (`networks/`): Neural network definitions
- **Training scripts** (`training/`): Training loops and utilities
- **Evaluation tools** (`evaluation/`): Evaluation and comparison scripts
- **Utility modules** (`utils/`): Configuration, plotting, device management
- **Scripts** (`scripts/`): Analysis, visualization, and experiment scripts
- **Environments** (`environments/`): Reward wrappers and environment setup

### Academic Integrity

This project adheres to academic integrity standards by:
- Clearly documenting all AI tool usage
- Providing detailed attribution for AI-generated code
- Ensuring all code is reviewed and understood by team members
- Maintaining transparency about the development process
- Following course guidelines for AI tool usage in academic work

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

