"""LightningModule for coordinate diffusion training.

Wraps CoordinateDiffusionModel with Lightning training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from .base import BaseCiffyModule
from ciffy.nn.config import OutputConfig, TrainingConfig, WandbConfig
from ciffy.nn.diffusion.coordinate_diffusion import (
    CoordinateDiffusionConfig,
    CoordinateDiffusionModel,
)

if TYPE_CHECKING:
    from ciffy import Polymer


@dataclass
class CoordinateDiffusionDataConfig:
    """Dataset configuration for coordinate diffusion training."""

    data_dir: str = ""
    batch_size: int = 8  # Smaller due to full coordinate tensors
    molecule_types: tuple[str, ...] = ("RNA",)
    min_atoms: int = 50
    max_atoms: int = 2000


@dataclass
class CoordinateDiffusionFullConfig:
    """Full configuration for coordinate diffusion training."""

    model: CoordinateDiffusionConfig = field(default_factory=CoordinateDiffusionConfig)
    data: CoordinateDiffusionDataConfig = field(
        default_factory=CoordinateDiffusionDataConfig
    )
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


class CoordinateDiffusionModule(BaseCiffyModule):
    """LightningModule for training coordinate diffusion models.

    Unlike latent diffusion, this module works directly with atomic
    coordinates. Due to variable atom counts per polymer, batches are
    processed as lists of Polymer objects rather than padded tensors.

    The module expects batches from CoordinateDiffusionDataModule:
    - List of Polymer objects

    Example:
        >>> config = CoordinateDiffusionFullConfig.from_yaml("config.yaml")
        >>> module = CoordinateDiffusionModule(config)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
    """

    def __init__(self, config: CoordinateDiffusionFullConfig) -> None:
        """Initialize the coordinate diffusion module.

        Args:
            config: Full training configuration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.training_config = config.training

        # Create the model
        self.model = CoordinateDiffusionModel(config.model)

    def _aggregate_polymer_losses(
        self, batch: list["Polymer"]
    ) -> tuple[torch.Tensor, int]:
        """Aggregate losses across a batch of polymers.

        Due to variable atom counts, we process each polymer individually
        and weight losses by atom count for fair averaging.

        Args:
            batch: List of Polymer objects from dataloader.

        Returns:
            Tuple of (total_weighted_loss, total_atoms).
        """
        total_loss = torch.tensor(0.0, device=self.device)
        total_atoms = 0

        for polymer in batch:
            if polymer is None:
                continue

            loss, _ = self.model.training_step(polymer)

            # Weight by number of atoms for fair averaging
            n_atoms = polymer.size()
            total_loss = total_loss + loss * n_atoms
            total_atoms += n_atoms

        return total_loss, total_atoms

    def training_step(
        self,
        batch: list["Polymer"],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute training loss for a batch of polymers.

        Args:
            batch: List of Polymer objects from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Aggregated loss tensor.
        """
        total_loss, total_atoms = self._aggregate_polymer_losses(batch)

        if total_atoms == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        avg_loss = total_loss / total_atoms
        self.log("train/loss", avg_loss, prog_bar=True, on_step=True, on_epoch=True)

        return avg_loss

    def validation_step(
        self,
        batch: list["Polymer"],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute validation loss for a batch.

        Args:
            batch: List of Polymer objects from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Aggregated loss tensor.
        """
        total_loss, total_atoms = self._aggregate_polymer_losses(batch)

        if total_atoms == 0:
            return torch.tensor(0.0, device=self.device)

        avg_loss = total_loss / total_atoms
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True)

        return avg_loss


__all__ = [
    "CoordinateDiffusionDataConfig",
    "CoordinateDiffusionFullConfig",
    "CoordinateDiffusionModule",
]
