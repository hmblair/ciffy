"""
Reusable training utilities for PyTorch models.

This module provides checkpoint and device utilities that work with
any PyTorch model. For training loops, use Fabric-based trainers directly.

Example:
    >>> from ciffy.nn.training import get_device, save_checkpoint, load_checkpoint
    >>>
    >>> device = get_device("auto")
    >>> model = MyModel().to(device)
    >>> # ... training ...
    >>> save_checkpoint("checkpoint.pt", model, optimizer, epoch=10)
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a training experiment.

    Stores metrics and metadata from a completed (or failed) training run,
    enabling comparison across multiple experiments.

    Attributes:
        name: Experiment identifier (typically config filename without extension).
        config_path: Path to the YAML configuration file.
        status: One of 'success', 'failed', or 'running'.
        final_loss: Loss value from the final epoch.
        best_loss: Best (lowest) loss achieved during training.
        recon_loss: Final reconstruction loss component (VAE).
        kl_loss: Final KL divergence loss component (VAE).
        epochs_trained: Number of epochs completed.
        total_epochs: Total epochs configured for training.
        n_samples: Total samples processed.
        device: Device used for training (e.g., 'cuda:0', 'cpu').
        duration_seconds: Total training time in seconds.
        checkpoint_path: Path to the final/best checkpoint file.
        log_file: Path to log file containing stdout/stderr from the experiment.
        error: Error message if status is 'failed', None otherwise.
    """

    name: str
    config_path: str
    status: str  # 'success', 'failed', 'running'
    final_loss: float | None = None
    best_loss: float | None = None
    recon_loss: float | None = None
    kl_loss: float | None = None
    epochs_trained: int = 0
    total_epochs: int = 0
    n_samples: int = 0
    device: str = ""
    duration_seconds: float = 0.0
    checkpoint_path: str | None = None
    log_file: str | None = None
    error: str | None = None


def get_device(
    requested: str = "auto",
    rank: Optional[int] = None,
) -> "torch.device":
    """
    Get the appropriate torch device with automatic fallbacks.

    Args:
        requested: Device string. Options:
            - ``"auto"``: Try cuda > mps > cpu
            - ``"cuda"``: CUDA GPU (fails if unavailable)
            - ``"mps"``: Apple Silicon GPU (fails if unavailable)
            - ``"cpu"``: CPU
            - Specific device like ``"cuda:0"`` or ``"cuda:1"``
        rank: Optional rank for distributed training. If provided with
            ``"cuda"``, selects ``cuda:{rank % num_gpus}``.

    Returns:
        torch.device object.

    Raises:
        RuntimeError: If requested device is not available.

    Example:
        >>> device = get_device("auto")
        >>> device = get_device("cuda", rank=0)  # For distributed training
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_device")

    if requested == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = requested

    # Handle distributed training with CUDA
    if device_str == "cuda" and rank is not None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("CUDA requested but no GPUs available")
        device_str = f"cuda:{rank % num_gpus}"

    # Validate device availability
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available, cannot use device '{device_str}'")
    elif device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS not available on this system")

    device = torch.device(device_str)
    logger.debug(f"Using device: {device}")
    return device


def save_checkpoint(
    path: str | Path,
    model: "nn.Module",
    optimizer: Optional["optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[dict[str, Any]] = None,
    config: Optional[Any] = None,
    **extra: Any,
) -> None:
    """
    Save training checkpoint with model state and metadata.

    Args:
        path: Output file path (.pt).
        model: PyTorch model to save.
        optimizer: Optional optimizer state to save.
        scheduler: Optional learning rate scheduler state to save.
        epoch: Current epoch number.
        step: Current global step.
        metrics: Optional metrics dictionary (loss, accuracy, etc.).
        config: Optional config object/dict to store. Dataclasses are
            automatically converted to dicts.
        **extra: Additional key-value pairs to include in checkpoint.

    Note:
        Creates parent directories automatically. For distributed training,
        only rank 0 should call this function.

    Example:
        >>> save_checkpoint(
        ...     "checkpoints/epoch_10.pt",
        ...     model, optimizer,
        ...     epoch=10,
        ...     metrics={"loss": 0.5},
        ...     config=config,
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for save_checkpoint")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        # Convert dataclass to dict if needed
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            checkpoint["config"] = dataclasses.asdict(config)
        elif hasattr(config, "__dict__"):
            checkpoint["config"] = vars(config)
        else:
            checkpoint["config"] = config

    # Add any extra data
    checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.debug(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    model: "nn.Module",
    optimizer: Optional["optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str | "torch.device"] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load training checkpoint and restore state.

    Args:
        path: Checkpoint file path (.pt).
        model: Model to load state into (modified in-place).
        optimizer: Optional optimizer to restore state (modified in-place).
        scheduler: Optional scheduler to restore state (modified in-place).
        map_location: Device to map tensors to (e.g., "cpu", "cuda:0").
            If None, tensors are loaded to the same device they were saved from.
        strict: If True, require exact key match for model state_dict.
            Set to False when loading partial weights.

    Returns:
        Checkpoint dict with metadata (epoch, step, metrics, config, etc.).
        Model/optimizer/scheduler are loaded in-place, not returned.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.

    Example:
        >>> ckpt = load_checkpoint("checkpoints/best.pt", model, optimizer)
        >>> start_epoch = ckpt["epoch"] + 1
        >>> best_loss = ckpt["metrics"]["loss"]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_checkpoint")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.debug(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


__all__ = [
    "ExperimentResult",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
]
