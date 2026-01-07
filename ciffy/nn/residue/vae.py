"""Residue-level VAE for polymer generative modeling."""

from __future__ import annotations

import torch
import torch.nn as nn

from ciffy.polymer import Polymer

from .encoder import ResidueEncoder
from .decoder import ResidueDecoder


class ResidueVAE(nn.Module):
    """
    Complete VAE for residue-level polymer modeling.

    Combines ResidueEncoder and ResidueDecoder into a full VAE that can:
    - Encode polymers to per-residue latent distributions
    - Decode latent samples to local coordinates and inter-residue transforms
    - Generate new polymer conformations via sampling

    Args:
        latent_dim: Dimension of the latent space per residue.
        d_model: Hidden dimension for transformer/MLP layers.
        n_heads: Number of attention heads in encoder.
        encoder_layers: Number of transformer layers in encoder.
        decoder_layers: Number of MLP layers in decoder.
        atom_dim: Dimension for atom type embeddings. Defaults to d_model // 2.
        residue_dim: Dimension for residue type embeddings. Defaults to d_model // 2.
        dropout: Dropout probability.

    Example:
        >>> model = ResidueVAE(latent_dim=32, d_model=128)
        >>> coords, transforms, mu, logvar = model(polymer)
        >>>
        >>> # Reconstruction
        >>> z = model.encode(polymer)
        >>> coords, transforms = model.decode(z, polymer)
        >>>
        >>> # Sampling
        >>> z = torch.randn(n_residues, 32)
        >>> coords, transforms = model.decode(z, template)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        atom_dim: int | None = None,
        residue_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ResidueEncoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
            dropout=dropout,
        )

        self.decoder = ResidueDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=decoder_layers,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
            dropout=dropout,
        )

    @property
    def latent_dim(self) -> int:
        """Latent dimension per residue."""
        return self.encoder.output_dim

    def encode(
        self,
        polymer: Polymer,
        transforms: torch.Tensor | None = None,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode polymer to per-residue latent vectors.

        Args:
            polymer: Input polymer (torch backend).
            transforms: Optional (n_residues, 6) inter-residue transforms.
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z: (n_residues, latent_dim) latent vectors.
            If return_distribution: (z, mu, logvar) tuple.
        """
        return self.encoder(polymer, transforms, return_distribution)

    def decode(
        self,
        z: torch.Tensor,
        polymer: Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latents to local coordinates and transforms.

        Args:
            z: (n_residues, latent_dim) latent vectors.
            polymer: Template polymer for atom structure.

        Returns:
            coords: (n_atoms, 3) local coordinates.
            transforms: (n_residues, 6) inter-residue SE(3) transforms.
        """
        return self.decoder(z, polymer)

    def forward(
        self,
        polymer: Polymer,
        transforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Args:
            polymer: Input polymer (torch backend).
            transforms: Optional (n_residues, 6) inter-residue transforms.

        Returns:
            coords: (n_atoms, 3) predicted local coordinates.
            pred_transforms: (n_residues, 6) predicted inter-residue transforms.
            mu: (n_residues, latent_dim) latent means.
            logvar: (n_residues, latent_dim) latent log-variances.
        """
        z, mu, logvar = self.encode(polymer, transforms, return_distribution=True)
        coords, pred_transforms = self.decode(z, polymer)
        return coords, pred_transforms, mu, logvar
