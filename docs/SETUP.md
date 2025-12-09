# Setup Instructions

This document explains how to set up the Rocket Lander RL project environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for neural networks)
- Gymnasium (for RL environments)
- NumPy (for numerical computations)
- Matplotlib (for plotting)
- Pandas (for data analysis)
- PyYAML (for configuration files)

### 2. GPU Support (Optional but Recommended)

#### macOS (Apple Silicon - M1/M2/M3)

For macOS with Apple Silicon chips, PyTorch can use Metal Performance Shaders (MPS) for GPU acceleration:

1. **Install PyTorch** (standard installation includes MPS support):
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Verify MPS availability**:
   ```python
   import torch
   print("PyTorch version:", torch.__version__)
   print("MPS available:", torch.backends.mps.is_available())  # Should print True on Apple Silicon
   print("MPS built:", torch.backends.mps.is_built())  # Should print True
   ```

3. **Note**: The code automatically detects and uses MPS when available. On macOS, `torch.cuda.is_available()` will be `False` (that's expected - CUDA is NVIDIA-only).

#### Linux/Windows with NVIDIA GPU (CUDA)

For systems with NVIDIA GPUs:

1. **Check CUDA availability**:
   ```bash
   nvidia-smi
   ```
   (Note: This command only works on Linux/Windows with NVIDIA GPUs, not on macOS)

2. **Install PyTorch with CUDA support**:
   Visit [PyTorch website](https://pytorch.org/get-started/locally/) and install the appropriate version for your system.

   For example, for CUDA 11.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Or for CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify GPU access**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print your GPU name
   ```

#### CPU-Only Installation

If you don't have a GPU or prefer CPU-only:
- The default PyTorch installation from `requirements.txt` will work on CPU
- Training will be slower but fully functional
- The code automatically falls back to CPU if no GPU is available

### 3. Verify Installation

Run a quick test to verify everything is installed correctly:

```python
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("MPS (Metal) available: True (Apple Silicon GPU)")
else:
    print("Using CPU")

env = gym.make("LunarLander-v3")
print("Environment created successfully!")
print("State space:", env.observation_space)
print("Action space:", env.action_space)
```

## Environment Configuration

### LunarLander-v3 Environment

The project uses Gymnasium's LunarLander-v3 environment, which requires Box2D.

**Installation:**

1. **Install SWIG** (required for Box2D):
   
   **On macOS (use Homebrew, NOT pip):**
   ```bash
   brew install swig
   ```
   
   **On Linux:**
   ```bash
   sudo apt-get install swig
   # or
   pip install swig
   ```

2. **Install Box2D Python bindings:**
   ```bash
   pip install box2d-py
   ```

3. **Install pygame (required for rendering):**
   ```bash
   pip install pygame
   ```
   
   Or install all at once:
   ```bash
   pip install box2d-py pygame
   ```

3. **Verify installation:**
   ```bash
   python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print('Success!'); env.close()"
   ```

**Note**: If Box2D installation fails, see [INSTALL_BOX2D.md](INSTALL_BOX2D.md) for detailed troubleshooting instructions, including:
- Platform-specific installation methods (macOS, Linux, Windows)
- Common error messages and solutions
- Alternative installation approaches
- Verification steps

### Custom Environments

The project includes:
- **Toy Rocket Environment**: A simple discrete grid-based environment (no additional dependencies)
- **Reward Wrapper**: Custom wrapper for LunarLander-v3

## Directory Structure

After running experiments, the following directories will be created:

```
cs372final/
├── checkpoints/      # Saved model checkpoints
├── plots/            # Generated plots and figures
├── logs/             # Training logs
├── results/          # Evaluation results (CSV files)
└── hyperparameter_results/  # Hyperparameter sweep results
```

These directories are created automatically when you run training scripts.

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Make sure you're in the project root directory
   - Check that all packages are installed: `pip list`

2. **GPU/CUDA Issues**:
   - **macOS**: CUDA is not available (NVIDIA-only). Use MPS instead (automatic on Apple Silicon)
   - **CUDA Out of Memory**: Reduce batch size in training scripts
   - The code automatically falls back to CPU if GPU is unavailable
   - If you see "No matching distribution found for torchaudio" on macOS, use: `pip install torch torchvision torchaudio` (without CUDA index)

3. **Box2D Installation Issues**:
   - See [INSTALL_BOX2D.md](INSTALL_BOX2D.md) for comprehensive troubleshooting guide
   - Common issue: SWIG must be installed via Homebrew on macOS, not pip
   - If LunarLander-v3 fails to import, check Box2D installation first

4. **Environment Not Found**:
   - Make sure gymnasium is installed: `pip install gymnasium`
   - Try: `python -c "import gymnasium as gym; gym.make('LunarLander-v3')"`
   - If Box2D errors occur, see [INSTALL_BOX2D.md](INSTALL_BOX2D.md)

4. **Plotting Issues**:
   - Make sure matplotlib is installed: `pip install matplotlib`
   - On headless servers, you may need: `pip install matplotlib --upgrade`

## Next Steps

After setup, you can:

1. Train a simple agent:
   ```bash
   python training/train_dqn.py --episodes 100
   ```

2. Run all experiments:
   ```bash
   python scripts/run_all_experiments.py
   ```

3. See [README.md](README.md) for more usage examples.

