"""LightningModule for residue VAE training.

Unlike flow models, VAE doesn't need PCA preprocessing - it learns
dimensionality reduction end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ciffy.nn.config import TrainingConfig
from .vae_base import BaseVAEModule

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.vae.residue.model import ResidueVAE


@dataclass
class ResidueVAEModelConfig:
    """Configuration for ResidueVAE model.

    The beta parameter controls the reconstruction-regularization trade-off:
    - Higher beta (1.0): Better N(0,1) latent but worse reconstruction
    - Lower beta (0.1): Better reconstruction but latent deviates from N(0,1)

    The free_bits parameter helps balance this by only applying KL penalty
    above a threshold per dimension, allowing minimum information usage.

    The gamma parameter controls geometry sampling loss - sampling from the
    prior and penalizing invalid geometry (bond lengths, angles). This helps
    the model generate chemically valid structures.
    """

    latent_dim: int = 12
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    beta: float = 1.0  # KL weight (standard VAE)
    beta_warmup_epochs: int = 50  # Epochs to linearly warm up beta
    free_bits: float = 0.5  # Min nats/dim before KL penalty - prevents collapse
    dropout: float = 0.0
    use_input_norm: bool = True  # Learn input normalization (improves reconstruction)
    use_residual: bool = True  # Residual connections in decoder
    separate_heads: bool = True  # Separate output heads for coords vs transforms
    gamma: float = 0.0  # Geometry sampling loss weight (0 = disabled)
    n_geom_samples: int = 16  # Number of samples for geometry loss


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


class ResidueVAEModule(BaseVAEModule):
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
        >>> # Works with PolymerModel!
        >>> from ciffy.nn import PolymerModel
        >>> polymer_model = PolymerModel({Residue.A: model, ...})
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
        super().__init__(
            model_config=config.model,
            training_config=config.training,
            residue=residue,
        )
        self.config = config

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

        atom_indices = atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms)

        # Create ResidueVAE
        self._residue_model = ResidueVAE(
            input_dim=n_features,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            residue=self.residue,
            atom_indices=atom_indices,
            dropout=config.dropout,
            use_input_norm=config.use_input_norm,
            use_residual=config.use_residual,
            separate_heads=config.separate_heads,
        )

        # Set up geometry sampling loss if gamma > 0
        self._setup_geometry_loss(atom_indices)

    def _forward_batch(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a batch.

        Args:
            batch: Tuple containing (data,) tensor of shape (batch_size, n_features).
            batch_idx: Batch index (unused).

        Returns:
            recon: Reconstructed output.
            target: Original data.
            mu: Latent means.
            logvar: Latent log-variances.
        """
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon, mu, logvar = self._residue_model(data)
        return recon, data, mu, logvar


__all__ = [
    "ResidueVAEModelConfig",
    "ResidueVAEDataConfig",
    "ResidueVAEFullConfig",
    "ResidueVAEModule",
]
