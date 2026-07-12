import numpy as np

def to_pytorch(array):
    """
    Zero-copy conversion bridge from Thermite/numpy arrays directly to PyTorch tensors.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install it to use this feature.")
    
    return torch.from_dlpack(array)

def to_jax(array):
    """
    Zero-copy conversion bridge from Thermite/numpy arrays directly to JAX arrays.
    """
    try:
        import jax.dlpack
    except ImportError:
        raise ImportError("JAX is not installed. Please install it to use this feature.")
    
    return jax.dlpack.from_dlpack(array)
