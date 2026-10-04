"""
Device configuration module for the inference system.
This module provides functions to set and get the default device for model execution.
"""

import model_bridge

def set_device(device: str) -> bool:
    """
    Set the default device for model execution.
    
    Args:
        device: The device to use ('cuda' or 'cpu')
        
    Returns:
        bool: True if the device was set successfully, False otherwise
    """
    return model_bridge.set_default_device(device)

def get_device() -> str:
    """
    Get the current default device for model execution.
    
    Returns:
        str: The current default device ('cuda' or 'cpu')
    """
    return model_bridge.get_default_device()

def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    # If the current device is 'cuda', then CUDA is available
    return get_device() == 'cuda'