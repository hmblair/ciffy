"""Learning rate scheduler utilities.

Provides factory functions for creating PyTorch LR schedulers with optional warmup.

Example:
    >>> from ciffy.nn import create_scheduler, SchedulerConfig
    >>> config = SchedulerConfig(scheduler_type="cosine", warmup_epochs=5, min_lr=1e-6)
    >>> scheduler = create_scheduler(optimizer, config, total_epochs=100)
    >>> for epoch in range(100):
    ...     train_epoch(...)
    ...     scheduler.step()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
        StepLR,
        _LRScheduler,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    optim = None
    _LRScheduler = None

if TYPE_CHECKING:
    from .base_trainer import SchedulerConfig


def create_scheduler(
    optimizer: "optim.Optimizer",
    config: "SchedulerConfig",
    total_epochs: int,
) -> "_LRScheduler | None":
    """Create LR scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer.
        config: SchedulerConfig instance.
        total_epochs: Total number of training epochs.

    Returns:
        LR scheduler, or None if scheduler_type is 'none'.

    Raises:
        ValueError: If scheduler_type is not recognized.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> config = SchedulerConfig(scheduler_type="cosine", warmup_epochs=5)
        >>> scheduler = create_scheduler(optimizer, config, total_epochs=100)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for LR scheduling")

    if config.scheduler_type == "none":
        return None

    warmup_epochs = config.warmup_epochs
    main_epochs = max(1, total_epochs - warmup_epochs)

    # Create main scheduler
    if config.scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=main_epochs,
            eta_min=config.min_lr,
        )
    elif config.scheduler_type == "linear":
        # Linear decay from initial LR to min_lr
        initial_lr = optimizer.param_groups[0]["lr"]
        end_factor = config.min_lr / initial_lr if initial_lr > 0 else 1.0
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=main_epochs,
        )
    elif config.scheduler_type == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {config.scheduler_type}. "
            f"Expected one of: 'cosine', 'linear', 'step', 'none'"
        )

    # Add warmup if configured
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler


def get_current_lr(scheduler: "_LRScheduler | None") -> float | None:
    """Get current learning rate from scheduler.

    Args:
        scheduler: LR scheduler, or None.

    Returns:
        Current learning rate, or None if no scheduler.
    """
    if scheduler is None:
        return None
    return scheduler.get_last_lr()[0]


__all__ = [
    "create_scheduler",
    "get_current_lr",
]
