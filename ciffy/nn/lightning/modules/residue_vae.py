"""LightningModule for residue VAE training.

Unlike flow models, VAE doesn't need PCA preprocessing - it learns
dimensionality reduction end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ciffy.nn.config import TrainingConfig

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.vae.residue.model import ResidueVAE


@dataclass
class ResidueVAEModelConfig:
    """Configuration for ResidueVAE model."""

    latent_dim: int = 12
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    beta: float = 1.0  # KL weight (beta-VAE)
    beta_warmup_epochs: int = 50  # Epochs to linearly warm up beta
    dropout: float = 0.0


@dataclass
class ResidueVAEDataConfig:
    """Data configuration for residue VAE training."""

    data_dir: str = ""
    cif_patterns: list[str] | None = None
    residue: str = "A"
    min_coverage: float = 0.9
    train_split: float = 0.8
    batch_size: int = 256


@dataclass
class ResidueVAEFullConfig:
    """Full configuration for residue VAE training."""

    model: ResidueVAEModelConfig = field(default_factory=ResidueVAEModelConfig)
    data: ResidueVAEDataConfig = field(default_factory=ResidueVAEDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


class ResidueVAEModule(LightningModule):
    """LightningModule for training residue VAE models.

    Unlike flow models, VAE doesn't need PCA preprocessing:
    - Encoder learns dimensionality reduction end-to-end
    - Uses ELBO loss (reconstruction + KL divergence)
    - Creates a full ResidueVAE with metadata for save/load

    Uses the same ResidueDataModule as flow models - the data format
    [coords_flat, transforms] works for both.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.lightning import ResidueVAEModule, ResidueDataModule
        >>> import lightning as L
        >>>
        >>> config = ResidueVAEFullConfig()
        >>> dm = ResidueDataModule(cif_paths, residue=Residue.A)
        >>> module = ResidueVAEModule(config, residue=Residue.A)
        >>>
        >>> trainer = L.Trainer(max_epochs=200)
        >>> trainer.fit(module, dm)
        >>>
        >>> # Get the trained model for inference/saving
        >>> model = module.get_model()
        >>> model.save("my_vae_model")
        >>>
        >>> # Works with PolymerFlowModel!
        >>> from ciffy.nn.flow import PolymerFlowModel
        >>> polymer_model = PolymerFlowModel({Residue.A: model, ...})
    """

    def __init__(
        self,
        config: ResidueVAEFullConfig,
        residue: "Residue",
    ) -> None:
        """Initialize the residue VAE module.

        Args:
            config: Full training configuration.
            residue: Residue type to train on.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["residue"])

        self.config = config
        self.training_config = config.training
        self.residue = residue

        # Model created in setup()
        self._residue_model: "ResidueVAE | None" = None

    @property
    def model(self) -> torch.nn.Module | None:
        """The underlying ResidueVAE model."""
        return self._residue_model

    def get_model(self) -> "ResidueVAE":
        """Get the trained ResidueVAE for inference/saving.

        Returns:
            The trained ResidueVAE with all metadata.

        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self._residue_model is None:
            raise ValueError("Model not yet created. Run trainer.fit() first.")
        return self._residue_model

    def setup(self, stage: str) -> None:
        """Create model from data module info.

        Unlike flow models, no PCA is needed - we just need to know
        the input dimension and atom indices.
        """
        if stage != "fit" or self._residue_model is not None:
            return

        from ciffy.nn.vae.residue.model import ResidueVAE

        # Get data info from datamodule
        dm = self.trainer.datamodule
        atoms = dm.atoms
        n_features = dm.n_features  # n_atoms * 3 + 6

        if atoms is None:
            raise ValueError("DataModule not set up properly - atoms is None")

        config = self.config.model

        # Create ResidueVAE
        self._residue_model = ResidueVAE(
            input_dim=n_features,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            residue=self.residue,
            atom_indices=atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms),
            dropout=config.dropout,
        )

    def _get_beta(self) -> float:
        """Get current beta value with optional warmup."""
        config = self.config.model
        if config.beta_warmup_epochs <= 0:
            return config.beta

        # Linear warmup
        warmup_progress = min(1.0, self.current_epoch / config.beta_warmup_epochs)
        return config.beta * warmup_progress

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute ELBO loss for a batch.

        Loss = reconstruction_loss + beta * KL_divergence

        Args:
            batch: Tuple containing (data,) tensor of shape (batch_size, n_features).
            batch_idx: Batch index (unused).

        Returns:
            ELBO loss.
        """
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Forward pass
        recon, mu, logvar = self._residue_model(data)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, data, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Per-sample, then mean across batch
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # ELBO with beta weighting
        beta = self._get_beta()
        loss = recon_loss + beta * kl_loss

        # Logging
        self.log("train/recon", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/kl", kl_loss, on_step=True, on_epoch=True)
        self.log("train/beta", beta, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train/z_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train/z_std", mu.std(), on_step=False, on_epoch=True)
        self.log("train/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation ELBO."""
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        recon, mu, logvar = self._residue_model(data)

        recon_loss = F.mse_loss(recon, data, reduction="mean")
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        beta = self._get_beta()
        loss = recon_loss + beta * kl_loss

        self.log("val/recon", recon_loss, sync_dist=True)
        self.log("val/kl", kl_loss, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        config = self.training_config

        optimizer = AdamW(
            self._residue_model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay if hasattr(config, "weight_decay") else 0.01,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=config.scheduler.min_lr if hasattr(config, "scheduler") else 1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping if configured."""
        if self.training_config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self._residue_model.parameters(),
                self.training_config.grad_clip,
            )


__all__ = [
    "ResidueVAEModelConfig",
    "ResidueVAEDataConfig",
    "ResidueVAEFullConfig",
    "ResidueVAEModule",
]
