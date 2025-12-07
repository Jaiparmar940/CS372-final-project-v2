"""
Device detection utilities for GPU/CPU support.
Automatically detects and uses CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
"""

import torch


def get_device():
    """
    Get the appropriate device (CUDA/MPS if available, else CPU).
    Priority: CUDA > MPS > CPU
    
    Returns:
        torch.device: The device to use for computations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU (CUDA): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (MPS/Metal) - Apple Silicon")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def to_device(tensor, device):
    """
    Move tensor to specified device.
    
    Args:
        tensor: Tensor or tensor-like object
        device: Target device
        
    Returns:
        Tensor on the specified device
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor

