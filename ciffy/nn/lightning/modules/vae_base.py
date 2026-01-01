"""Base class for VAE training modules.

Provides common functionality for different VAE architectures:
- Beta scheduling with warmup
- KL divergence computation with free bits
- Consistent logging interface
- Optimizer and scheduler configuration
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ciffy.nn.config import TrainingConfig
from ciffy.nn.vae.losses import compute_kl_divergence, get_beta_with_warmup

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


@runtime_checkable
class VAEModelConfig(Protocol):
    """Protocol for VAE model configurations."""

    beta: float
    beta_warmup_epochs: int
    free_bits: float


@dataclass
class BaseVAEModelConfig:
    """Base configuration for VAE models.

    Subclass this for specific VAE architectures.
    """

    latent_dim: int = 12
    beta: float = 1.0
    beta_warmup_epochs: int = 50
    free_bits: float = 0.5


class BaseVAEModule(LightningModule):
    """Base class for VAE training modules.

    Subclasses must implement:
    - setup(): Create the model
    - _forward_batch(): Forward pass returning (recon, mu, logvar)
    - get_model(): Return the trained model

    Provides:
    - Beta scheduling with warmup
    - KL computation with free bits
    - Training/validation step logic
    - Optimizer configuration
    """

    def __init__(
        self,
        model_config: VAEModelConfig,
        training_config: TrainingConfig,
        residue: "Residue",
    ) -> None:
        """Initialize base VAE module.

        Args:
            model_config: Model configuration (must have beta, beta_warmup_epochs, free_bits).
            training_config: Training configuration.
            residue: Residue type to train on.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["residue"])

        self.model_config = model_config
        self.training_config = training_config
        self.residue = residue

    @abstractmethod
    def setup(self, stage: str) -> None:
        """Create model from data module info."""
        pass

    @abstractmethod
    def _forward_batch(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a batch.

        Args:
            batch: Batch from dataloader.
            batch_idx: Batch index.

        Returns:
            recon: Reconstructed output.
            target: Target output.
            mu: Latent means.
            logvar: Latent log-variances.
        """
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Get the trained model for inference/saving."""
        pass

    def get_beta(self) -> float:
        """Get current beta value with warmup."""
        return get_beta_with_warmup(
            self.current_epoch,
            self.model_config.beta,
            self.model_config.beta_warmup_epochs,
        )

    def compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with free bits."""
        return compute_kl_divergence(
            mu, logvar,
            free_bits=self.model_config.free_bits,
        )

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute ELBO loss for training."""
        recon, target, mu, logvar = self._forward_batch(batch, batch_idx)

        # Losses
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = self.compute_kl(mu, logvar)
        beta = self.get_beta()
        loss = recon_loss + beta * kl_loss

        # Logging
        self.log("train/recon", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/kl", kl_loss, on_step=True, on_epoch=True)
        self.log("train/beta", beta, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/z_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train/z_std", mu.std(), on_step=False, on_epoch=True)
        self.log("train/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation ELBO."""
        recon, target, mu, logvar = self._forward_batch(batch, batch_idx)

        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = self.compute_kl(mu, logvar)
        beta = self.get_beta()
        loss = recon_loss + beta * kl_loss

        self.log("val/recon", recon_loss, sync_dist=True)
        self.log("val/kl", kl_loss, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        config = self.training_config

        # Get model parameters - subclass must have _model or model attribute
        model = getattr(self, "_model", None) or getattr(self, "_residue_model", None)
        if model is None:
            raise RuntimeError("No model found. Ensure setup() creates _model or _residue_model.")

        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=getattr(config, "weight_decay", 0.01),
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=getattr(getattr(config, "scheduler", None), "min_lr", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping if configured."""
        if self.training_config.grad_clip:
            model = getattr(self, "_model", None) or getattr(self, "_residue_model", None)
            if model is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.grad_clip,
                )


__all__ = [
    "VAEModelConfig",
    "BaseVAEModelConfig",
    "BaseVAEModule",
]
