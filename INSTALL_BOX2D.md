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

### Method 1: Using Windows Package Manager (winget) - Recommended

```powershell
# Step 1: Install SWIG using Windows Package Manager
winget install --id SWIG.SWIG -e

# Step 2: Refresh PATH in current session (or restart terminal)
$env:Path += ";C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\SWIG.SWIG_Microsoft.Winget.Source_8wekyb3d8bbwe\swigwin-4.4.0"

# Step 3: Verify SWIG is accessible
swig -version

# Step 4: Install Box2D Python bindings
pip install box2d-py

# Step 5: Install pygame (required for rendering)
pip install pygame
```

**Note**: After installing SWIG via winget, you may need to restart your terminal or manually add SWIG to your PATH for the current session. The PATH will be automatically updated for new terminal sessions.

### Method 2: Manual Installation

```bash
# Install SWIG from https://www.swig.org/download.html
# Download swigwin-4.x.x.zip, extract it, and add to PATH

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
- **Windows**: Install SWIG using `winget install --id SWIG.SWIG -e` (recommended) or download manually from https://www.swig.org/download.html
- **macOS**: Install SWIG via Homebrew: `brew install swig` (NOT pip - the pip package is just a wrapper)
- **Linux**: Install via package manager: `sudo apt-get install swig`
- Make sure SWIG is in your PATH. On Windows, you may need to restart your terminal after installation.

## Alternative: Use Pre-trained Models

If Box2D installation continues to fail, you can:
1. Use the tabular Q-learning agent on the toy environment (no Box2D needed)
2. Train on a system where Box2D installs successfully
3. Use the evaluation scripts with pre-trained checkpoints

