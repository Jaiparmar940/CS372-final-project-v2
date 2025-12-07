"""
Device detection utilities for GPU/CPU support.
Automatically detects and uses CUDA when available, falls back to CPU otherwise.
"""

import torch


def get_device():
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device: The device to use for computations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
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

