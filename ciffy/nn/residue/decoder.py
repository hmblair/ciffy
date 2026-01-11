"""Residue-level decoder for polymer coordinate generation."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ciffy import Scale
from ciffy.nn import PolymerEmbedding
from ciffy.nn.layers.transformer import RMSNorm
from ciffy.polymer import Polymer


class ResidualBlock(nn.Module):
    """Residual block with pre-norm and SiLU activation."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = nn.functional.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


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
        rotation_repr: Rotation representation: "axis_angle" (6D output) or
            "rotation_6d" (9D output with continuous 6D rotation).

    Example:
        >>> decoder = ResidueDecoder(latent_dim=32, d_model=128)
        >>> coords, transforms = decoder(z, polymer)
        >>> # coords: (n_atoms, 3) local coordinates
        >>> # transforms: (n_residues, 6 or 9) inter-residue SE(3) transforms
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
        rotation_repr: Literal["axis_angle", "rotation_6d"] = "axis_angle",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.rotation_repr = rotation_repr

        # Output dimension: 6 for axis-angle, 9 for 6D rotation
        self.transform_dim = 9 if rotation_repr == "rotation_6d" else 6

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

        # Transform decoder (residue-level) with deeper residual blocks
        self.residue_embedding = PolymerEmbedding(
            scale=Scale.RESIDUE,
            residue_dim=d_model // 4,
        )

        # Input projection
        self.transform_proj = nn.Sequential(
            nn.Linear(latent_dim + self.residue_embedding.output_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks (4 blocks for deeper transform prediction)
        n_transform_blocks = 4
        self.transform_blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(n_transform_blocks)
        ])

        # Output head
        self.transform_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, self.transform_dim),
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
            transforms: (n_residues, 6 or 9) inter-residue SE(3) transforms.
                If rotation_repr="axis_angle": (axis-angle rotation, translation).
                If rotation_repr="rotation_6d": (6D rotation, translation).
        """
        # Expand latents to atom level
        z_expanded = polymer.expand(z, Scale.RESIDUE)
        atom_emb = self.embedding(polymer)

        # Decode coordinates
        x = torch.cat([z_expanded, atom_emb], dim=-1)
        x = self.coord_decoder_proj(x)
        coords = self.coord_head(x)

        # Decode transforms with residual blocks
        res_emb = self.residue_embedding(polymer)
        t = torch.cat([z, res_emb], dim=-1)
        t = self.transform_proj(t)
        for block in self.transform_blocks:
            t = block(t)
        transforms = self.transform_head(t)

        return coords, transforms
