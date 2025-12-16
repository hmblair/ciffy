"""
Reusable training utilities for PyTorch models.

This module provides function-based training primitives that work with
any PyTorch model and PolymerDataset. Designed for single-sample training
(batch_size=1) with DataLoader integration for shuffling and multiprocessing.

Example:
    >>> from ciffy.nn import PolymerDataset, PolymerVAE
    >>> from ciffy.nn.training import (
    ...     set_seed, get_device, train_epoch,
    ...     save_checkpoint, polymer_collate_fn, get_worker_init_fn,
    ... )
    >>> from torch.utils.data import DataLoader
    >>>
    >>> set_seed(42)
    >>> device = get_device("auto")
    >>> dataset = PolymerDataset("./data")
    >>> loader = DataLoader(
    ...     dataset, batch_size=1, shuffle=True,
    ...     collate_fn=polymer_collate_fn,
    ...     worker_init_fn=get_worker_init_fn(42),
    ... )
    >>> model = PolymerVAE().to(device)
    >>> optimizer = torch.optim.AdamW(model.parameters())
    >>>
    >>> def loss_fn(model, polymer):
    ...     return model.compute_loss(polymer.to(device))
    >>>
    >>> metrics = train_epoch(model, loader, loss_fn, optimizer)
"""

from __future__ import annotations

import dataclasses
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    np = None
    torch = None
    nn = None
    optim = None
    DataLoader = None

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
        deterministic: If True, enable PyTorch deterministic mode. This makes
            operations reproducible but may reduce performance. Affects cudnn
            and CUDA operations.

    Note:
        For distributed training, each rank should use ``seed + rank`` for
        different random states while maintaining reproducibility.

    Example:
        >>> set_seed(42)
        >>> # For distributed training:
        >>> set_seed(42 + rank)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for set_seed")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


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


def train_epoch(
    model: "nn.Module",
    dataloader: "DataLoader",
    loss_fn: Callable[["nn.Module", Any], dict[str, "torch.Tensor"]],
    optimizer: "optim.Optimizer",
    *,
    device: Optional["torch.device"] = None,
    grad_clip: Optional[float] = None,
    scheduler: Optional[Any] = None,
    step_scheduler_per_batch: bool = False,
    progress_bar: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> dict[str, float]:
    """
    Train model for one epoch with flexible loss function.

    This is a generic training loop that works with any model and loss function.
    The loss function receives the model and a single sample, and returns a dict
    containing at least a "loss" key.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader yielding samples. Use ``batch_size=1`` for
            variable-length polymers with ``polymer_collate_fn``.
        loss_fn: Callable with signature ``(model, sample) -> dict``.
            Must return dict with at least ``"loss"`` key containing a scalar
            tensor. May include other metrics like ``"recon_loss"``, ``"kl_loss"``.
        optimizer: Optimizer for parameter updates.
        device: Target device. If None, no device transfer is performed
            (loss_fn should handle device placement).
        grad_clip: Max gradient norm for clipping. None to disable.
        scheduler: Optional learning rate scheduler.
        step_scheduler_per_batch: If True, step scheduler after each batch.
            If False (default), scheduler should be stepped after epoch.
        progress_bar: If True, show tqdm progress bar.
        rank: Process rank for distributed training. Progress bar only
            shown on rank 0.
        world_size: Total number of processes for distributed training.

    Returns:
        Dict of averaged metrics over the epoch. Always includes:
        - ``"n_samples"``: Number of successfully processed samples
        - ``"n_skipped"``: Number of skipped samples (None or errors)
        Plus any keys returned by loss_fn (e.g., ``"loss"``, ``"recon_loss"``).

    Example:
        >>> def vae_loss_fn(model, polymer):
        ...     polymer = polymer.poly().to(device)
        ...     return model.compute_loss(polymer)
        >>>
        >>> metrics = train_epoch(model, loader, vae_loss_fn, optimizer)
        >>> print(f"Loss: {metrics['loss']:.4f}, Samples: {metrics['n_samples']}")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_epoch")

    model.train()
    metrics_accum: dict[str, float] = defaultdict(float)
    n_samples = 0
    n_skipped = 0

    # Progress bar (only rank 0 in distributed)
    show_progress = progress_bar and (rank is None or rank == 0)
    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(dataloader, desc="Training")
    else:
        pbar = dataloader

    for sample in pbar:
        # Skip None samples (filtered by collate_fn)
        if sample is None:
            n_skipped += 1
            continue

        try:
            # Move sample to device if specified and sample supports it
            if device is not None and hasattr(sample, "to"):
                sample = sample.to(device)

            # Compute loss
            optimizer.zero_grad()
            losses = loss_fn(model, sample)
            loss = losses["loss"]

            # Skip NaN losses
            if torch.isnan(loss):
                n_skipped += 1
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            optimizer.step()

            # Scheduler step (per-batch)
            if scheduler is not None and step_scheduler_per_batch:
                scheduler.step()

            # Accumulate metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    metrics_accum[key] += value.item()
                else:
                    metrics_accum[key] += value
            n_samples += 1

            # Update progress bar
            if show_progress and TQDM_AVAILABLE:
                pbar.set_postfix(
                    {
                        "loss": f"{metrics_accum['loss']/n_samples:.4f}",
                        "n": n_samples,
                        "skip": n_skipped,
                    }
                )

        except Exception as e:
            logger.debug(f"Skipping sample due to error: {e}")
            n_skipped += 1
            continue

    # Average metrics
    result: dict[str, float] = {}
    if n_samples > 0:
        for key, value in metrics_accum.items():
            result[key] = value / n_samples
    result["n_samples"] = float(n_samples)
    result["n_skipped"] = float(n_skipped)

    return result


def polymer_collate_fn(batch: list[Any]) -> Any | None:
    """
    Collate function for PolymerDataset that filters None values.

    Since we use ``batch_size=1`` for variable-length polymers, this function
    simply returns the single item or None if the batch is empty/invalid.

    Args:
        batch: List of samples from dataset (typically length 1).

    Returns:
        Single sample if valid, None if all samples were None.

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=1,
        ...     collate_fn=polymer_collate_fn,
        ... )
    """
    # Filter out None values
    valid = [x for x in batch if x is not None]
    if not valid:
        return None
    # batch_size=1: return single item
    return valid[0]


def get_worker_init_fn(base_seed: Optional[int] = None) -> Callable[[int], None]:
    """
    Get worker initialization function for DataLoader multiprocessing.

    Ensures each worker has a different but reproducible random seed,
    preventing all workers from generating the same random sequences.

    Args:
        base_seed: Base seed value. Each worker gets ``base_seed + worker_id``.
            If None, workers use arbitrary seeds (non-reproducible).

    Returns:
        Worker init function to pass to ``DataLoader(worker_init_fn=...)``.

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=get_worker_init_fn(seed=42),
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_worker_init_fn")

    def worker_init_fn(worker_id: int) -> None:
        if base_seed is not None:
            seed = base_seed + worker_id
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    return worker_init_fn


__all__ = [
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "train_epoch",
    "polymer_collate_fn",
    "get_worker_init_fn",
]
