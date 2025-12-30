"""LightningModule for latent diffusion training.

Wraps LatentDiffusionModel with Lightning training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from .base import BaseCiffyModule
from ciffy.nn.config import OutputConfig, TrainingConfig, WandbConfig
from ciffy.nn.diffusion.latent_diffusion import (
    LatentDiffusionConfig,
    LatentDiffusionModel,
)

if TYPE_CHECKING:
    from ciffy.nn.flow import PolymerFlowModel


@dataclass
class LatentDiffusionDataConfig:
    """Dataset configuration for latent diffusion training.

    Attributes:
        data_dir: Directory containing CIF files.
        batch_size: Training batch size.
        molecule_types: Filter to specific molecule types (e.g., ["RNA"]).
        min_residues: Minimum residues per chain.
        max_residues: Maximum residues per chain.
    """

    data_dir: str = ""
    batch_size: int = 32
    molecule_types: tuple[str, ...] = ("RNA",)
    min_residues: int = 10
    max_residues: int = 500


@dataclass
class LatentDiffusionFullConfig:
    """Full configuration for latent diffusion training.

    Combines model, data, training, output, and logging configurations.
    """

    model: LatentDiffusionConfig = field(default_factory=LatentDiffusionConfig)
    data: LatentDiffusionDataConfig = field(default_factory=LatentDiffusionDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # EMA settings
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000

    # Validation settings
    val_every: int = 10
    val_samples: int = 5
    val_steps: int = 50


class LatentDiffusionModule(BaseCiffyModule):
    """LightningModule for training latent diffusion models.

    This module wraps a LatentDiffusionModel and provides:
    - Training step with pre-encoded latent batches
    - Validation step with same loss computation
    - Configurable optimizer and scheduler via BaseCiffyModule

    The module expects batches from LatentDiffusionDataModule:
    - (latents, sequences, mask) tuple
    - latents: (batch, n_residues, latent_dim)
    - sequences: (batch, n_residues)
    - mask: (batch, n_residues) bool, True = padding

    Example:
        >>> config = LatentDiffusionFullConfig.from_yaml("config.yaml")
        >>> module = LatentDiffusionModule(config)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self,
        config: LatentDiffusionFullConfig,
        flow_model: Optional["PolymerFlowModel"] = None,
    ) -> None:
        """Initialize the latent diffusion module.

        Args:
            config: Full training configuration.
            flow_model: Optional pre-loaded flow model. If None, loads from
                config.model.flow_model_path or uses default pretrained.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["flow_model"])

        self.config = config
        self.training_config = config.training

        # Create the model
        self.model = LatentDiffusionModel(config.model, flow_model=flow_model)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute training loss for a batch.

        Args:
            batch: Tuple of (latents, sequences, mask) from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Loss tensor.
        """
        latents, sequences, mask = batch

        loss, metrics = self.model.training_step_batch(latents, sequences, mask)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/noise_mse", metrics["noise_mse"], on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute validation loss for a batch.

        Args:
            batch: Tuple of (latents, sequences, mask) from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Loss tensor.
        """
        latents, sequences, mask = batch

        loss, metrics = self.model.training_step_batch(latents, sequences, mask)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/noise_mse", metrics["noise_mse"], sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer for denoiser only (flow model is frozen)."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        config = self.training_config

        # Only optimize denoiser parameters (flow model is frozen)
        optimizer = AdamW(
            self.model.denoiser.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scheduler_config = config.scheduler
        if scheduler_config.scheduler_type == "none":
            return {"optimizer": optimizer}

        # Calculate epochs
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
        else:
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=main_epochs, eta_min=scheduler_config.min_lr
            )

        # Add warmup
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


__all__ = [
    "LatentDiffusionDataConfig",
    "LatentDiffusionFullConfig",
    "LatentDiffusionModule",
]
