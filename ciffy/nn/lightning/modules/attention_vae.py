"""LightningModule for attention-based residue VAE training.

The attention encoder naturally handles missing atoms via masking,
while always decoding to the full canonical atom set.
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
    from ciffy.nn.vae.residue.attention import AttentionResidueVAE


@dataclass
class AttentionVAEModelConfig:
    """Configuration for AttentionResidueVAE model.

    The attention encoder handles missing atoms naturally via masking.
    The MLP decoder always outputs the full canonical atom set.
    """

    latent_dim: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    beta: float = 1.0  # KL weight
    beta_warmup_epochs: int = 50
    free_bits: float = 0.5  # Min nats/dim before KL penalty


@dataclass
class AttentionVAEDataConfig:
    """Data configuration for attention VAE training."""

    data_dir: str = ""
    cif_patterns: list[str] | None = None
    residue: str = "A"
    min_coverage: float = 0.9
    train_split: float = 0.8
    batch_size: int = 256


@dataclass
class AttentionVAEFullConfig:
    """Full configuration for attention VAE training."""

    model: AttentionVAEModelConfig = field(default_factory=AttentionVAEModelConfig)
    data: AttentionVAEDataConfig = field(default_factory=AttentionVAEDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


class AttentionResidueVAEModule(BaseVAEModule):
    """LightningModule for training attention-based residue VAE.

    Uses attention encoder to handle variable/missing atoms (future support),
    while always decoding to the full canonical atom set for structure
    prediction.

    Currently works with ResidueDataModule which provides complete residues.
    Future versions can support partial residue training.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.lightning import AttentionResidueVAEModule, ResidueDataModule
        >>> import lightning as L
        >>>
        >>> config = AttentionVAEFullConfig()
        >>> dm = ResidueDataModule(cif_paths, residue=Residue.A)
        >>> module = AttentionResidueVAEModule(config, residue=Residue.A)
        >>>
        >>> trainer = L.Trainer(max_epochs=200)
        >>> trainer.fit(module, dm)
        >>>
        >>> # Get trained model
        >>> model = module.get_model()
        >>> model.save("my_attention_vae")
    """

    def __init__(
        self,
        config: AttentionVAEFullConfig,
        residue: "Residue",
    ) -> None:
        """Initialize the attention VAE module.

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
        self._model: "AttentionResidueVAE | None" = None
        self._n_atoms: int | None = None

    @property
    def model(self) -> torch.nn.Module | None:
        """The underlying AttentionResidueVAE model."""
        return self._model

    def get_model(self) -> "AttentionResidueVAE":
        """Get the trained model for inference/saving.

        Returns:
            The trained AttentionResidueVAE with all metadata.

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

        from ciffy.nn.vae.residue.attention import AttentionResidueVAE

        # Get data info from datamodule
        dm = self.trainer.datamodule
        atoms = dm.atoms
        n_features = dm.n_features  # n_atoms * 3 + 6

        if atoms is None:
            raise ValueError("DataModule not set up properly - atoms is None")

        self._n_atoms = (n_features - 6) // 3
        atom_indices = atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms)
        n_atom_types = max(atom_indices) + 1

        config = self.config.model

        # Create AttentionResidueVAE
        self._model = AttentionResidueVAE(
            n_atom_types=n_atom_types,
            n_atoms=self._n_atoms,
            latent_dim=config.latent_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            decoder_hidden_dims=config.decoder_hidden_dims,
            residue=self.residue,
            atom_indices=atom_indices,
            dropout=config.dropout,
        )

    def _unpack_batch(
        self, batch: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack flattened batch to coords, transforms, mask.

        The existing data module returns flattened [coords, transforms].
        We reshape to (batch, n_atoms, 3) and create all-ones mask.

        Args:
            batch: Tuple containing flattened data tensor.

        Returns:
            coords: (batch, n_atoms, 3)
            transforms: (batch, 6)
            mask: (batch, n_atoms) all True
        """
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        n_coord_dims = self._n_atoms * 3
        coords_flat = data[:, :n_coord_dims]
        transforms = data[:, n_coord_dims:]

        coords = coords_flat.reshape(-1, self._n_atoms, 3)
        mask = torch.ones(
            coords.shape[0], self._n_atoms,
            dtype=torch.bool, device=coords.device
        )

        return coords, transforms, mask

    def _forward_batch(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a batch.

        Args:
            batch: Tuple containing flattened data tensor.
            batch_idx: Batch index (unused).

        Returns:
            recon: Reconstructed output (flattened coords + transforms).
            target: Original data (flattened coords + transforms).
            mu: Latent means.
            logvar: Latent log-variances.
        """
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        coords, transforms_target, mask = self._unpack_batch(batch)

        # Forward pass
        recon_coords, recon_transforms, mu, logvar = self._model(coords, mask)

        # Flatten for base class
        recon = torch.cat([recon_coords.reshape(-1, self._n_atoms * 3), recon_transforms], dim=-1)

        return recon, data, mu, logvar


__all__ = [
    "AttentionVAEModelConfig",
    "AttentionVAEDataConfig",
    "AttentionVAEFullConfig",
    "AttentionResidueVAEModule",
]
