"""
Fabric utilities for training.

Provides a thin wrapper around PyTorch Lightning Fabric for training loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.fabric import Fabric


def create_fabric(
    device: str = "auto",
    precision: str = "32-true",
    strategy: str = "auto",
    num_devices: int = 1,
) -> "Fabric":
    """
    Create a configured Fabric instance for training.

    Args:
        device: Device to use. Options:
            - "auto": Automatically select GPU if available, else CPU
            - "cpu": Force CPU
            - "cuda": Use CUDA GPU (first available)
            - "cuda:0", "cuda:1": Specific GPU
            - "mps": Apple Silicon GPU
        precision: Training precision. Options:
            - "32-true": Full 32-bit precision (default)
            - "16-mixed": Mixed precision with 16-bit
            - "bf16-mixed": Mixed precision with bfloat16
        strategy: Distributed strategy. Options:
            - "auto": Automatic selection
            - "ddp": Distributed Data Parallel
            - "fsdp": Fully Sharded Data Parallel
        num_devices: Number of devices to use. Default 1 for single-GPU.
            Set higher for multi-GPU DDP training.

    Returns:
        Configured Fabric instance.

    Example:
        >>> fabric = create_fabric(device="cuda", precision="16-mixed")
        >>> fabric.launch()
        >>> model, optimizer = fabric.setup(model, optimizer)
    """
    from lightning.fabric import Fabric

    # Parse device string to accelerator and devices
    devices: int | list[int] = num_devices

    if device == "auto":
        accelerator = "auto"
        devices = num_devices
    elif device == "cpu":
        accelerator = "cpu"
        devices = 1  # CPU doesn't benefit from multiple "devices"
    elif device.startswith("cuda"):
        accelerator = "cuda"
        # Handle cuda:N format to select specific GPU
        if ":" in device:
            try:
                gpu_id = int(device.split(":")[1])
                devices = [gpu_id]
            except (ValueError, IndexError):
                devices = num_devices
        else:
            devices = num_devices
    elif device == "mps":
        accelerator = "mps"
        devices = 1  # MPS is single-device
    else:
        accelerator = "auto"
        devices = num_devices

    return Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
    )


__all__ = ["create_fabric"]
