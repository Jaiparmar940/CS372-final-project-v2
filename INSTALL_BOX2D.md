# Installing Box2D for LunarLander-v3

LunarLander-v3 requires Box2D, which can be tricky to install on some systems, especially macOS.

## macOS Installation

### Method 1: Homebrew + pip (Recommended - This Works!)

```bash
# Step 1: Install SWIG binary via Homebrew (NOT pip!)
brew install swig

# Step 2: Install Box2D Python bindings
pip install box2d-py

# Step 3: Install pygame (required for rendering)
pip install pygame
```

**Important**: 
- Install SWIG via Homebrew, not pip. The pip `swig` package is a Python wrapper, not the actual SWIG binary needed for compilation.
- pygame is required for visual rendering of LunarLander environments.

### Method 2: Direct Install

```bash
# Install SWIG first
pip install swig

# Try installing gymnasium with box2d
pip install "gymnasium[box2d]"
```

### Method 3: If Methods 1 & 2 Fail

```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install SWIG
pip install swig

# Try with no build isolation
pip install box2d-py --no-build-isolation
```

### Method 4: Using Conda (if available)

```bash
conda install -c conda-forge box2d-py
```

## Linux Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install swig libbox2d-dev

# Install Python package
pip install box2d-py
```

## Windows Installation

```bash
# Install SWIG from https://www.swig.org/download.html
# Add SWIG to PATH

# Then install
pip install box2d-py
```

## Verify Installation

After installation, verify it works:

```bash
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print('Success!'); env.close()"
```

## Troubleshooting

### "Failed to build box2d-py"
- Make sure SWIG is installed: `swig -version`
- On macOS, try installing Box2D via Homebrew first
- Try using a virtual environment with a fresh Python installation

### "ModuleNotFoundError: No module named 'Box2D'"
- Box2D installation failed - try a different method above
- Check that you're using the same Python interpreter where you installed Box2D

### "Command 'swig' failed"
- Install SWIG: `pip install swig` or `brew install swig` (macOS)
- Make sure SWIG is in your PATH

## Alternative: Use Pre-trained Models

If Box2D installation continues to fail, you can:
1. Use the tabular Q-learning agent on the toy environment (no Box2D needed)
2. Train on a system where Box2D installs successfully
3. Use the evaluation scripts with pre-trained checkpoints

