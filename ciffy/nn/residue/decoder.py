"""Residue-level decoder for polymer coordinate generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ciffy import Scale
from ciffy.nn import PolymerEmbedding
from ciffy.nn.layers.transformer import RMSNorm
from ciffy.polymer import Polymer


class ResidueDecoder(nn.Module):
    """
    Decodes per-residue latents to local coordinates and inter-residue transforms.

    Takes per-residue latent vectors and a template polymer, then predicts:
    - Local coordinates for each atom (in the residue's aligned frame)
    - Inter-residue SE(3) transforms for chain assembly

    Args:
        latent_dim: Dimension of the input latent space per residue.
        d_model: Hidden dimension for decoder layers.
        n_layers: Number of decoder layers.
        n_heads: Number of attention heads.
        atom_dim: Dimension for atom type embeddings. Defaults to d_model // 2.
        residue_dim: Dimension for residue type embeddings. Defaults to d_model // 2.
        dropout: Dropout probability.

    Example:
        >>> decoder = ResidueDecoder(latent_dim=32, d_model=128)
        >>> coords, transforms = decoder(z, polymer)
        >>> # coords: (n_atoms, 3) local coordinates
        >>> # transforms: (n_residues, 6) inter-residue SE(3) transforms
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        atom_dim: int | None = None,
        residue_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # Default embedding dimensions
        if atom_dim is None:
            atom_dim = d_model // 2
        if residue_dim is None:
            residue_dim = d_model // 2

        self._atom_dim = atom_dim
        self._residue_dim = residue_dim

        # Atom embeddings (same structure as encoder)
        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
        )
        embed_dim = self.embedding.output_dim

        # Coordinate decoder projection
        self.coord_decoder_proj = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # MLP decoder for coordinates
        coord_layers = []
        for _ in range(n_layers - 1):
            coord_layers.extend([
                nn.Linear(d_model, d_model),
                RMSNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        coord_layers.append(nn.Linear(d_model, 3))
        self.coord_head = nn.Sequential(*coord_layers)

        # Transform decoder (residue-level)
        self.residue_embedding = PolymerEmbedding(
            scale=Scale.RESIDUE,
            residue_dim=d_model // 4,
        )
        self.transform_decoder = nn.Sequential(
            nn.Linear(latent_dim + self.residue_embedding.output_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6),
        )

    def forward(
        self,
        z: torch.Tensor,
        polymer: Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latents to local coordinates and inter-residue transforms.

        Args:
            z: (n_residues, latent_dim) latent vectors.
            polymer: Template polymer for atom structure.

        Returns:
            coords: (n_atoms, 3) local coordinates (in residue-aligned frames).
            transforms: (n_residues, 6) inter-residue SE(3) transforms
                       (axis-angle rotation, translation).
        """
        # Expand latents to atom level
        z_expanded = polymer.expand(z, Scale.RESIDUE)
        atom_emb = self.embedding(polymer)

        # Decode coordinates
        x = torch.cat([z_expanded, atom_emb], dim=-1)
        x = self.coord_decoder_proj(x)
        coords = self.coord_head(x)

        # Decode transforms
        res_emb = self.residue_embedding(polymer)
        t_input = torch.cat([z, res_emb], dim=-1)
        transforms = self.transform_decoder(t_input)

        return coords, transforms
