"""LightningModule for consolidated VAE training.

Trains a ConsolidatedResidueVAE with shared encoder and per-residue decoders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ciffy.nn.config import TrainingConfig
from ciffy.nn.vae.losses import compute_kl_divergence, get_beta_with_warmup, GeometrySamplingLoss

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.vae.residue.consolidated import ConsolidatedResidueVAE


@dataclass
class ConsolidatedVAEModelConfig:
    """Configuration for ConsolidatedResidueVAE model."""

    latent_dim: int = 12
    d_model: int = 64
    d_dist: int = 32
    n_heads: int = 4
    n_encoder_layers: int = 2
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    beta: float = 1.0
    beta_warmup_epochs: int = 50
    free_bits: float = 0.5
    use_input_norm: bool = True  # Learn input normalization (improves reconstruction)
    use_residual: bool = True  # Residual connections in decoder
    gamma: float = 0.0  # Geometry sampling loss weight (0 = disabled)
    n_geom_samples: int = 16  # Number of samples for geometry loss


@dataclass
class ConsolidatedVAEDataConfig:
    """Data configuration for consolidated VAE training."""

    residues: str = "ACGU"
    min_coverage: float = 0.9
    train_split: float = 0.8
    batch_size: int = 256


@dataclass
class ConsolidatedVAEFullConfig:
    """Full configuration for consolidated VAE training."""

    model: ConsolidatedVAEModelConfig = field(default_factory=ConsolidatedVAEModelConfig)
    data: ConsolidatedVAEDataConfig = field(default_factory=ConsolidatedVAEDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


class ConsolidatedVAEModule(LightningModule):
    """LightningModule for training ConsolidatedResidueVAE.

    The consolidated VAE has a shared encoder for all residue types and
    separate decoder heads for each type. This allows learning shared
    representations while handling different atom counts.

    Training batches contain mixed residue types. The forward pass groups
    samples by residue type for proper decoder selection.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.lightning.data import ConsolidatedDataModule
        >>> import lightning as L
        >>>
        >>> config = ConsolidatedVAEFullConfig()
        >>> dm = ConsolidatedDataModule(cif_paths, residues=[Residue.A, ...])
        >>> module = ConsolidatedVAEModule(config, residues=[Residue.A, ...])
        >>>
        >>> trainer = L.Trainer(max_epochs=200)
        >>> trainer.fit(module, dm)
        >>>
        >>> # Get trained model
        >>> model = module.get_model()
    """

    def __init__(
        self,
        config: ConsolidatedVAEFullConfig,
        residues: list["Residue"],
    ) -> None:
        """Initialize the consolidated VAE module.

        Args:
            config: Full training configuration.
            residues: List of residue types to train on.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["residues"])

        self.config = config
        self.residues = residues

        # Model created in setup()
        self._model: "ConsolidatedResidueVAE | None" = None

        # Geometry sampling loss (set in setup if gamma > 0)
        self._geometry_loss: GeometrySamplingLoss | None = None

    def get_model(self) -> "ConsolidatedResidueVAE":
        """Get the trained ConsolidatedResidueVAE.

        Returns:
            The trained model.

        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self._model is None:
            raise ValueError("Model not yet created. Run trainer.fit() first.")
        return self._model

    def setup(self, stage: str) -> None:
        """Create model from data module info."""
        if stage != "fit" or self._model is not None:
            return

        from ciffy.nn.vae.residue.consolidated import (
            ConsolidatedResidueVAE,
            ConsolidatedVAEConfig,
        )

        # Get residue_atoms from datamodule
        dm = self.trainer.datamodule
        residue_atoms = dm.residue_atoms

        if residue_atoms is None:
            raise ValueError("DataModule not set up properly - residue_atoms is None")

        model_cfg = self.config.model

        config = ConsolidatedVAEConfig(
            latent_dim=model_cfg.latent_dim,
            d_model=model_cfg.d_model,
            d_dist=model_cfg.d_dist,
            n_heads=model_cfg.n_heads,
            n_encoder_layers=model_cfg.n_encoder_layers,
            hidden_dims=model_cfg.hidden_dims,
            dropout=model_cfg.dropout,
            use_input_norm=model_cfg.use_input_norm,
            use_residual=model_cfg.use_residual,
        )

        self._model = ConsolidatedResidueVAE(residue_atoms, config)

        # Set up geometry sampling loss if gamma > 0
        if model_cfg.gamma > 0:
            self._geometry_loss = GeometrySamplingLoss.from_residue_atoms(residue_atoms)

    def get_beta(self) -> float:
        """Get current beta value with warmup."""
        return get_beta_with_warmup(
            self.current_epoch,
            self.config.model.beta,
            self.config.model.beta_warmup_epochs,
        )

    def compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with free bits."""
        return compute_kl_divergence(
            mu, logvar,
            free_bits=self.config.model.free_bits,
        )

    def _forward_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a batch with mixed residue types.

        Encodes all samples together (shared encoder), then decodes per residue type.
        Uses vectorized operations where possible.

        Returns:
            coord_loss: Scalar coordinate reconstruction loss.
            transform_loss: Scalar transform reconstruction loss.
            mu: (batch, latent_dim) latent means.
            logvar: (batch, latent_dim) latent log-variances.
        """
        atom_types, coords, masks, transforms, residue_idxs = batch
        device = atom_types.device
        batch_size = coords.shape[0]

        # Encode all samples together (shared encoder)
        z, mu, logvar = self._model.encode_batch(
            atom_types, coords, masks, transforms=transforms, return_distribution=True
        )

        # Pre-compute masks for all residue types (vectorized)
        n_residues = len(self.residues)
        residue_masks = residue_idxs.unsqueeze(1) == torch.arange(n_residues, device=device)
        # residue_masks: (batch, n_residues) - True where sample matches residue type

        # Accumulate losses
        coord_losses = []
        transform_losses = []

        for res_idx, res in enumerate(self.residues):
            # Get samples for this residue type using pre-computed mask
            sample_indices = residue_masks[:, res_idx].nonzero(as_tuple=True)[0]
            if len(sample_indices) == 0:
                continue

            # Gather samples (vectorized)
            z_res = z[sample_indices]
            coords_res = coords[sample_indices]
            transforms_res = transforms[sample_indices]

            # Decode
            recon_coords, recon_transforms = self._model.decode(z_res, res)
            n_atoms = recon_coords.shape[1]

            # Compute per-sample losses (vectorized)
            target_coords = coords_res[:, :n_atoms, :]
            coord_loss = (recon_coords - target_coords).pow(2).mean(dim=(1, 2))  # (n_samples,)
            transform_loss = (recon_transforms - transforms_res).pow(2).mean(dim=1)  # (n_samples,)

            coord_losses.append(coord_loss)
            transform_losses.append(transform_loss)

        # Concatenate and average across all samples
        if coord_losses:
            coord_loss = torch.cat(coord_losses).mean()
            transform_loss = torch.cat(transform_losses).mean()
        else:
            coord_loss = torch.tensor(0.0, device=device)
            transform_loss = torch.tensor(0.0, device=device)

        return coord_loss, transform_loss, mu, logvar

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Compute ELBO loss for training."""
        coord_loss, transform_loss, mu, logvar = self._forward_batch(batch)

        # Combined reconstruction loss (coords + transforms)
        recon_loss = coord_loss + transform_loss

        # Losses
        kl_loss = self.compute_kl(mu, logvar)
        beta = self.get_beta()
        loss = recon_loss + beta * kl_loss

        # Geometry sampling loss
        gamma = self.config.model.gamma
        if gamma > 0 and self._geometry_loss is not None:
            n_samples = self.config.model.n_geom_samples
            geom_loss = self._geometry_loss.compute(self._model, n_samples)
            loss = loss + gamma * geom_loss
            self.log("train/geom", geom_loss, on_step=True, on_epoch=True)

        # Logging
        self.log("train/coord_loss", coord_loss, on_step=True, on_epoch=True)
        self.log("train/transform_loss", transform_loss, on_step=True, on_epoch=True)
        self.log("train/recon", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/kl", kl_loss, on_step=True, on_epoch=True)
        self.log("train/beta", beta, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Compute validation ELBO."""
        coord_loss, transform_loss, mu, logvar = self._forward_batch(batch)

        # Combined reconstruction loss
        recon_loss = coord_loss + transform_loss

        kl_loss = self.compute_kl(mu, logvar)
        beta = self.get_beta()
        loss = recon_loss + beta * kl_loss

        self.log("val/coord_loss", coord_loss, sync_dist=True)
        self.log("val/transform_loss", transform_loss, sync_dist=True)
        self.log("val/recon", recon_loss, sync_dist=True)
        self.log("val/kl", kl_loss, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        config = self.config.training

        optimizer = AdamW(
            self._model.parameters(),
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
        if self.config.training.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.config.training.grad_clip,
            )


__all__ = [
    "ConsolidatedVAEModelConfig",
    "ConsolidatedVAEDataConfig",
    "ConsolidatedVAEFullConfig",
    "ConsolidatedVAEModule",
]
