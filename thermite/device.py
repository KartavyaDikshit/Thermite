"""
thermite.device - Device selection utilities.

Users select the compute backend via a string argument:
    device='cpu'   -> DeviceKind::Cpu  (default, always available)
    device='gpu'   -> DeviceKind::Gpu  (wgpu, dispatches to best GPU)
    device='cuda'  -> DeviceKind::Gpu  (alias for wgpu on NVIDIA)

The thermite-gpu Rust crate handles dispatch transparently.
If the wgpu feature is not compiled in, GPU calls silently fall back to CPU.
"""

DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
DEVICE_CUDA = "cuda"

_VALID_DEVICES = {DEVICE_CPU, DEVICE_GPU, DEVICE_CUDA}


def validate_device(device: str) -> str:
    """Validate and normalise a device string."""
    d = device.lower().strip()
    if d not in _VALID_DEVICES:
        raise ValueError(
            f"Unknown device '{device}'. Valid options: {sorted(_VALID_DEVICES)}"
        )
    return d


def is_gpu(device: str) -> bool:
    """Return True if device resolves to GPU backend."""
    return validate_device(device) in (DEVICE_GPU, DEVICE_CUDA)
