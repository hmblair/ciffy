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
) -> "Fabric":
    """
    Create a configured Fabric instance for training.

    Args:
        device: Device to use. Options:
            - "auto": Automatically select GPU if available, else CPU
            - "cpu": Force CPU
            - "cuda": Use CUDA GPU
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

    Returns:
        Configured Fabric instance.

    Example:
        >>> fabric = create_fabric(device="cuda", precision="16-mixed")
        >>> fabric.launch()
        >>> model, optimizer = fabric.setup(model, optimizer)
    """
    from lightning.fabric import Fabric

    # Parse device string to accelerator
    if device == "auto":
        accelerator = "auto"
    elif device == "cpu":
        accelerator = "cpu"
    elif device.startswith("cuda"):
        accelerator = "cuda"
    elif device == "mps":
        accelerator = "mps"
    else:
        accelerator = "auto"

    return Fabric(
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
    )


__all__ = ["create_fabric"]
