"""Base LightningModule for ciffy models.

Provides shared training logic for optimizer configuration, learning rate
scheduling, gradient clipping, and structure validation metrics.
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
    from ciffy.polymer import Polymer
    from ciffy.geometry.constraints import GeometryConstraints


class BaseCiffyModule(LightningModule):
    """Base LightningModule with shared training configuration.

    Subclasses should:
    - Set `self.model` to the model being trained (must have `compute_loss(polymer)`)
    - Set `self.training_config` to training hyperparameters
    - Optionally override `training_step()` if custom batch handling is needed
    - Optionally implement `validation_step()`

    This base class provides:
    - Default `training_step()` that calls `model.compute_loss(polymer)`
    - Optimizer creation (AdamW)
    - Learning rate scheduling (cosine, linear, step, with optional warmup)
    - Gradient clipping
    - Structure validation metric helpers

    Example:
        >>> class MyModule(BaseCiffyModule):
        ...     def __init__(self, model, training_config):
        ...         super().__init__()
        ...         self.model = model  # Must have compute_loss(polymer)
        ...         self.training_config = training_config
        ...
        >>> # That's it! training_step() is inherited from BaseCiffyModule
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

    def training_step(self, polymer: "Polymer", batch_idx: int) -> torch.Tensor:
        """Default training step for models with compute_loss(polymer).

        Assumes:
        - `self.model` has a `compute_loss(polymer)` method
        - DataLoader yields individual Polymer objects

        Override this method if your model needs different batch handling.

        Args:
            polymer: Input polymer from DataLoader.
            batch_idx: Batch index (unused but required by Lightning).

        Returns:
            Loss tensor for backpropagation.
        """
        loss = self.model.compute_loss(polymer.strip())
        self.log("train/loss", loss, prog_bar=True)
        return loss

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

    # =========================================================================
    # Structure Validation Metrics
    # =========================================================================

    def log_structure_metrics(
        self,
        pred: "Polymer",
        target: "Polymer",
        prefix: str = "val",
        log_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Compute and log standard structure comparison metrics.

        Logs RMSD, TM-score, and radius of gyration for predicted vs target.
        Call this in your validation_step() for automatic metric logging.

        Args:
            pred: Predicted polymer structure.
            target: Ground truth polymer structure.
            prefix: Logging prefix (e.g., "val", "test").
            log_kwargs: Additional kwargs for self.log() (e.g., batch_size).

        Returns:
            Dict of computed metrics for further processing.

        Example:
            >>> def validation_step(self, batch, batch_idx):
            ...     pred = self.model.sample(batch.template)[0]
            ...     metrics = self.log_structure_metrics(pred, batch.target)
            ...     return metrics["rmsd"]
        """
        from ciffy.operations.metrics import rmsd, tm_score, rg

        log_kwargs = log_kwargs or {}
        metrics = {}

        # RMSD (Kabsch-aligned)
        try:
            rmsd_val = float(rmsd(pred, target).mean())
            metrics["rmsd"] = rmsd_val
            self.log(f"{prefix}/rmsd", rmsd_val, **log_kwargs)
        except Exception:
            pass  # Skip if structures incompatible

        # TM-score
        try:
            tm_val = tm_score(pred, target)
            metrics["tm_score"] = tm_val
            self.log(f"{prefix}/tm_score", tm_val, **log_kwargs)
        except Exception:
            pass  # Skip if representative atoms missing

        # Radius of gyration (pred only - measures compactness)
        try:
            rg_pred = float(rg(pred).mean())
            rg_target = float(rg(target).mean())
            metrics["rg_pred"] = rg_pred
            metrics["rg_target"] = rg_target
            metrics["rg_diff"] = abs(rg_pred - rg_target)
            self.log(f"{prefix}/rg", rg_pred, **log_kwargs)
            self.log(f"{prefix}/rg_diff", metrics["rg_diff"], **log_kwargs)
        except Exception:
            pass

        return metrics

    def log_geometry_metrics(
        self,
        coords: torch.Tensor,
        transforms: torch.Tensor | None = None,
        constraints: "GeometryConstraints | None" = None,
        prefix: str = "val",
        log_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Compute and log geometry constraint metrics (bonds, angles).

        Uses GeometryConstraints to validate bond lengths and angles.
        Call this in validation_step() for residue-level models.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.
            transforms: Optional (batch, 6) inter-residue transforms.
            constraints: GeometryConstraints instance. If None, skips.
            prefix: Logging prefix (e.g., "val", "test").
            log_kwargs: Additional kwargs for self.log().

        Returns:
            Dict of computed metrics.

        Example:
            >>> def validation_step(self, batch, batch_idx):
            ...     coords, transforms = self.model(batch)
            ...     metrics = self.log_geometry_metrics(
            ...         coords, transforms, self.constraints
            ...     )
        """
        if constraints is None:
            return {}

        log_kwargs = log_kwargs or {}
        metrics = {}

        # Use GeometryConstraints.compute_error_metrics()
        if transforms is None:
            transforms = torch.zeros(coords.shape[0], 6, device=coords.device)

        error_metrics = constraints.compute_error_metrics(coords, transforms)

        for key, value in error_metrics.items():
            metrics[key] = value
            self.log(f"{prefix}/{key}", value, **log_kwargs)

        return metrics

    def log_clash_metrics(
        self,
        polymer: "Polymer",
        prefix: str = "val",
        log_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Compute and log steric clash metrics.

        Args:
            polymer: Structure to check for clashes.
            prefix: Logging prefix.
            log_kwargs: Additional kwargs for self.log().

        Returns:
            Dict with clash count and clash fraction.
        """
        from ciffy.operations.metrics import clashes

        log_kwargs = log_kwargs or {}
        metrics = {}

        try:
            clash_pairs = clashes(polymer)
            n_clashes = len(clash_pairs)
            n_atoms = polymer.size()
            clash_frac = n_clashes / max(1, n_atoms * (n_atoms - 1) / 2)

            metrics["n_clashes"] = n_clashes
            metrics["clash_frac"] = clash_frac
            self.log(f"{prefix}/n_clashes", float(n_clashes), **log_kwargs)
            self.log(f"{prefix}/clash_frac", clash_frac, **log_kwargs)
        except Exception:
            pass

        return metrics


__all__ = ["BaseCiffyModule"]
