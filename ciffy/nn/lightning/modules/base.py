"""Base LightningModule for ciffy models.

Provides shared training logic for optimizer configuration, learning rate
scheduling, and gradient clipping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)

if TYPE_CHECKING:
    from ciffy.nn.config import TrainingConfig


class BaseCiffyModule(LightningModule):
    """Base LightningModule with shared training configuration.

    Subclasses should:
    - Set `self.model` to the model being trained
    - Set `self.training_config` to training hyperparameters
    - Implement `training_step()` and optionally `validation_step()`

    This base class handles:
    - Optimizer creation (AdamW)
    - Learning rate scheduling (cosine, linear, step, with optional warmup)
    - Gradient clipping

    Example:
        >>> class MyModule(BaseCiffyModule):
        ...     def __init__(self, config):
        ...         super().__init__()
        ...         self.model = MyModel(config.model)
        ...         self.training_config = config.training
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         loss = self.model(batch)
        ...         self.log("train/loss", loss)
        ...         return loss
    """

    # Subclasses should set these
    model: torch.nn.Module
    training_config: "TrainingConfig"

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Uses settings from `self.training_config`:
        - lr: Learning rate
        - weight_decay: L2 regularization
        - scheduler.scheduler_type: 'cosine', 'linear', 'step', or 'none'
        - scheduler.warmup_epochs: Warmup period
        - scheduler.min_lr: Minimum learning rate

        Returns:
            Dictionary with 'optimizer' and optionally 'lr_scheduler'.
        """
        config = self.training_config

        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Check if scheduler is needed
        scheduler_config = config.scheduler
        if scheduler_config.scheduler_type == "none":
            return {"optimizer": optimizer}

        # Calculate total epochs (accounting for warmup)
        total_epochs = self.trainer.max_epochs
        warmup_epochs = scheduler_config.warmup_epochs
        main_epochs = max(1, total_epochs - warmup_epochs)

        # Create main scheduler
        if scheduler_config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=scheduler_config.min_lr,
            )
        elif scheduler_config.scheduler_type == "linear":
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=scheduler_config.min_lr / config.lr,
                total_iters=main_epochs,
            )
        elif scheduler_config.scheduler_type == "step":
            main_scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma,
            )
        else:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_config.scheduler_type}"
            )

        # Add warmup if configured
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping before optimizer step.

        Uses `self.training_config.grad_clip` if set.
        Logs gradient norms if `self.training_config.log_gradient_norms` is True.
        """
        if not hasattr(self, "training_config"):
            return

        config = self.training_config

        # Log gradient norm before clipping (if enabled)
        log_norms = getattr(config, "log_gradient_norms", False)
        if log_norms:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("train/grad_norm_raw", total_norm, on_step=True, on_epoch=False)

        # Apply gradient clipping
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                config.grad_clip,
            )
            # Log clipped norm
            if log_norms:
                clipped_norm = min(total_norm, config.grad_clip)
                self.log("train/grad_norm_clipped", clipped_norm, on_step=True, on_epoch=False)


__all__ = ["BaseCiffyModule"]
